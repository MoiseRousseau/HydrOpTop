import h5py
import numpy as np
import matplotlib.pyplot as plt
import meshio

class IO:
  r"""
  Default initializer of the Input/Output class.
    
  By default, output optimization results (i.e. density parameter `p`) in file ``HydrOpTop.vtu`` and keep a trace of the history of the cost function and constraints in ``HydrOpTop.txt``.
  
  """
  def __init__(self, filename="HydrOpTop", logfile="out.txt", fileformat="vtu"):
    self.output_filename = filename
    self.output_log_name = logfile 
    self.output_format = fileformat
    self.output_file = None
    
    self.cf_name = "cf"
    
    self.output_every = 0
    self.output_number = 0
    self.output_gradient_obj = False
    self.output_grad_constraints = False
    self.output_initial = True
    self.output_mat_props = False
    self.output_adj_obj = False
    self.initialized = False
    
    self.vertices = None
    self.elements = None
    self.var_loc = None
    return
  
  
  def define_output_file(self, filename):
    r"""
    Define the output file name (without its extension) where to save the optimization results.
    Extension are automatically added based on output file format chosen (see method ``define_output_format``)
    
    :param filename: The output filename (default=``"HydrOpTop"`` )
    :type filename: str
    """
    self.output_filename = filename
    return
  
  
  def define_output_format(self, f):
    r"""
    Set the output format.
        
    :param f: Format of the output file. Must be one of ``["med", "vtu", "xdmf"]``
    :type f: str
    """
    self.output_format = f
    return
  
  
  def define_output_log(self, filename="out.txt"):
    """
    Define the output file name (with extension) where the history of the cost function value and constraints are stored.

    :param filename: output log path
    :type filename: str
    """
    self.output_log_name = filename
    return
  
  
  def no_output_initial(self):
    """
    By default, HydrOpTop output the initial density parameters and other programmed variables. This command cancel this behavior.
    """
    self.output_initial = False
    return
  
  
  def output_every_iteration(self, n):
    r"""
    Specify to output the density parameter :math:`p` every :math:`n` iteration in file and format specified in ``define_output_file()`` and ``define_output_format()`` methods.
    
    :param n: the iteration interval, ex. ``n=2`` for every two iteration.
    :type n: int
    """
    self.output_every = n
    return
  
  
  def output_gradient(self):
    r"""
    Enable writing the gradient of the objective function relative to the density parameter :math:`p` in the output file.
    """
    self.output_gradient_obj = True
    return
  
  
  def output_gradient_constrains(self):
    r"""
    Enable writing the gradient of the constraints relative to the density parameter :math:`p`.
    """
    self.output_grad_constraints = True
    return
  
  
  def output_adjoint_objective(self):
    """
    Enable writing the gradient of the constraints relative to the density parameter :math:`p`.
    """
    self.output_adj_obj = True
    return


  def output_material_properties(self):
    r"""
    Enable writing the material properties at parametrized cells.
    """
    self.output_mat_props = True
    return
  
  
  def communicate_functions_names(self, cf, constrains):
    self.cf_name = cf
  
  def output(self, it, #iteration number
             cf, #cost function value
             constraints_val, #constraints value
             p_raw=None, #raw density parameter (not filtered)
             grad_cf=None, # d(cost function) / d p
             grad_constraints=None, # d(constraints) / d p
             mat_props=None, # parametrized mat props (dict)
             p_filtered=None, #filtered density parameters
             adj_obj=None,
             val_at=None, # cell/node ids corresponding to dataset
             final=False): # Final value to output
    if not self.initialized: self.initiate_output(constraints_val) 
    
    #output to log
    out = open(self.output_log_name, 'a')
    out.write(f"{it}\t{cf:.6e}")
    for c in constraints_val.values():
      out.write(f"\t{c:.6e}")
    out.write("\n")
    out.close()

    if not final:
      if (not self.output_number) and (not self.output_initial):
        return
      if (self.output_every == 0 or \
         (it % self.output_every) != 0) and (self.output_number):
        return

      if not self.output_number:
        if not self.output_initial:
          self.output_number += 1
          return
    
    #output field
    dict_var = {}
    #add raw density parameter p
    dict_var["Density parameter"] = self.correct_dataset_length(
      p_raw, self.var_loc["Density parameter"], val_at
    )
    #add filtered density parameter p
    if p_filtered is not None:
      dict_var["Density parameter filtered"] = self.correct_dataset_length(
        p_filtered, self.var_loc["Density parameter filtered"],
        val_at
      )
    #add gradient
    if self.output_gradient_obj:
      dict_var[f"Gradient d{self.cf_name}_dp"] = self.correct_dataset_length(
        grad_cf,
        self.var_loc["Density parameter"],
        val_at
      )
    #add gradient constraint
    if self.output_grad_constraints:
      for name,grad in grad_constraints.items():
        dict_var[f"Gradient d{name}_dp"] = self.correct_dataset_length(
          grad,
          self.var_loc["Density parameter"],
          val_at
        )
    #add mat props
    if self.output_mat_props:
      for mat_name, X in mat_props.items():
        dict_var[mat_name] = self.correct_dataset_length(
          X, self.var_loc[mat_name], val_at
        )
    # add adjoint vector
    if self.output_adj_obj:
      dict_var[f"Adjoint vector"] = adj_obj
    
    point_var = {k:v for k,v in dict_var.items() if self.var_loc[k] == "point"}
    cell_var = {k:v for k,v in dict_var.items() if self.var_loc[k] == "cell"}
    self.outputter.write(cell_var, point_var, self.output_number)
    self.output_number += 1
    return
    
  def correct_dataset_length(self, X, var_loc, val_at=None):
    """
    Correct X_dataset if not of the size of the mesh.
    Usable when parametrized cells is lower than the whole mesh
    If X not linked to val_at (different size), output NaN
    val_at 0 based
    """
    if val_at is None: 
      return X
    if var_loc == "cell":
      X_new = np.zeros(self.n_elements, dtype='f8')
    else:
      X_new = np.zeros(len(self.vertices), dtype='f8')
    X_new[:] = np.nan
    if len(val_at) == len(X):
      X_new[val_at] = X
    return X_new
    
  def initiate_output(self, constraints_val):
    self.initialized = True
    #initiate log
    out = open(self.output_log_name, 'w')
    out.write(f"Iteration\t{self.cf_name}")
    for name in constraints_val.keys():
      out.write(f"\t{name}")
    out.write('\n')
    out.close()
    #initiate output
    if self.output_format == "xdmf_native":
      self.outputter = IO_XDMF(self.output_filename)
    if self.output_format == "xdmf":
      self.outputter = IO_XDMF_MESHIO(self.output_filename)
    else:
      self.outputter = IO_MESHIO(self.output_filename, self.output_format)
    self.outputter.set_mesh(self.vertices, self.elements, 
                            self.indexes)
    return
  
  def set_mesh_info(self, vertices, elements, indexes):
    #use by the crafter to pass mesh information
    self.vertices = vertices
    self.elements = elements
    self.n_elements = 0
    for (elem_type, elems) in elements:
      self.n_elements += len(elems)
    self.indexes = indexes
    return
  
  def communicate_var_location(self, var_loc):
    self.var_loc = var_loc
    
  def write_fields_to_file(self, X, filename, Xname, var_loc="cell", at_ids=None):
    r"""
    Output the field data given in the list X using ``MeshIO`` python library.
    
    :param X: The list of field datas to output
    :type X: list of numpy array
    :param filename: The name of the output file. Note, the format is deduced from the file extension.
    :type param: str
    :param Xname: The dataset names. Must be ordered the same as X.
    :type Xname: list of str
    :param var_loc: Location of the field (cell or point)
    :type var_loc: str
    :param at_ids: If the X datasets does not span the whole simulation domain, the `at_ids`` array give the point/cell ids corresponding to the given data.
    :type: iterable (same size as X)
    """
    X_ = []
    for x in X:
      if at_ids is not None:
        x_ = self.correct_dataset_length(x, var_loc, at_ids)
      else:
        x_ = x
      X_.append(x_)
    dict_var = {}
    for i,x in enumerate(X_):
      data = []
      for (elem_type, index) in self.indexes:
        data.append(x[index])
      dict_var[Xname[i]] = data
    if var_loc == "cell":
      mesh = meshio.Mesh(self.vertices, self.elements, cell_data=dict_var)
    elif var_loc == "point":
      mesh = meshio.Mesh(self.vertices, self.elements, point_data=dict_var)
    mesh.write(filename)
    return
  
  
  def plot_convergence_history(self, include_constraints=False, save_fig_to=None):
    r"""
    Description:
      Plot the convergence history of the optimization (cost function and constraints)
    
    Parameters:
      ``include_constraints`` (bool): Visualize evolution of constraints values (default: ``False``)
    
      ``save_fig_to`` (str): Save figure to the given file (optional, default: not saved but showed)
      
    |
    
    """
    data = np.genfromtxt(self.output_log_name, delimiter='\t', skip_header=1)
    src = open(self.output_log_name, 'r')
    header = src.readline()[:-1].split('\t')
    src.close()
    it = data[:,0]
    fig,ax = plt.subplots()
    ax.plot(it, data[:,1], label=header[1], color='r')
    ax.set_xlabel("# iterations")
    ax.set_xlim([0,it[-1]])
    ax.set_ylabel(f"{header[1]}")
    if np.max(data[:,1]) / np.min(data[:,1]) > 30.:
      ax.set_yscale("log")
    ax.grid()
    ax.legend()
    
    if include_constraints and len(header) > 2:
      ax2 = ax.twinx()
      for i in range(len(header)-2):
        ax2.plot(it, data[:,i+2], label=header[i+2])
      ax2.set_ylabel("Constraints")
      ax2.legend()

    plt.tight_layout()
    if save_fig_to:
      plt.savefig()
    else:
      plt.show()

    return
  
    

class IO_XDMF_MESHIO:
  #TODO: not working right now
  def __init__(self, f):
    self.filename = f
    self.vertices = None
    self.elements = None
    self.mesh_writed = False #is mesh already write ?
    return 
    
  def read(self):
    return
    
  def write(self, dict_var, n_output):
    #transform dict_var
    for var, data in dict_var.items():
      new_data = []
      for (elem_type, index) in self.indexes:
        new_data.append(data[index])
      dict_var[var] = new_data
    if not self.mesh_writed:
      self.mesh_writed = True
      self.writer.write_points_cells(self.vertices, self.elements)
    if self.var_loc == "cell":
      self.writer.write_data(n_output, cell_data=dict_var)
    elif self.var_loc == "point":
      self.writer.write_data(n_output, point_data=dict_var)
    self.writer.__exit__()
    return
  
  def set_mesh(self, vertices, elements, indexes, var_loc):
    self.vertices = vertices
    self.elements = elements
    self.indexes = indexes
    self.var_loc = var_loc
    self.writer = meshio.xdmf.TimeSeriesWriter(self.filename)
    self.writer.__enter__()
    return



class IO_MESHIO:
  def __init__(self, f, format_):
    self.n_output = 0
    self.filename = f
    corresp = {"vtu":"vtu", "medit":"mesh", "medit_binary":"meshb", "med":"med", "cgns":"cgns"}
    self.fileformat = corresp[format_]
    self.vertices = None
    self.elements = None
    return 
    
  def read(self, f):
    return
    
  def write(self, cell_var, point_var, n_output):
    for var, data in cell_var.items():
      new_data = []
      for (elem_type, index) in self.indexes:
        new_data.append(data[index])
      cell_var[var] = new_data
    mesh = meshio.Mesh(
      self.vertices,
      self.elements,
      point_data=point_var,
      cell_data=cell_var
    )
    out = self.filename + '-' + str(n_output) + "." + self.fileformat
    print(f"Output to {out}")
    mesh.write(out)
    return
  
  def set_mesh(self, vertices, elements, indexes):
    self.vertices = vertices
    self.elements = elements
    self.indexes = indexes
    return


class IO_PYVISTA:
  # TODO
  def __init__(self, f):
    return
    
  def read(self, f):
    return
    
  def write(self, cell_var, point_var, n_output):
    return
  
  def set_mesh(self, vertices, elements, indexes):
    return
