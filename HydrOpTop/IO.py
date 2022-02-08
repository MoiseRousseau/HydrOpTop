import h5py
import numpy as np
import meshio

class IO:
  """
  !Main class for input output
  Input:
  @param filename: output file for density parameter
  @param fileformat: output file format (xdmf, vtk)
  @param logfile: log filename (for cost function value, constrain, ...)
  """
  def __init__(self, filename="HydrOpTop", logfile="out.txt", fileformat="vtu"):
    self.output_filename = filename
    self.output_log_name = logfile 
    self.output_format = fileformat
    self.output_file = None
    
    self.cf_name = "cf"
    self.constrains_names = ["constrain"]
    
    self.output_every = 0
    self.output_number = 0
    self.output_gradient_obj = False
    self.output_grad_constraints = False
    self.output_initial = True
    self.initialized = False
    
    self.vertices = None
    self.elements = None
    self.var_loc = None
    
    return
  
  def define_output_file(self,f):
    self.output_filename = f
    return
  
  def define_output_format(self, f):
    self.output_format = f
    return
  
  def define_output_log(self, f):
    self.output_log_name = f
    return
  
  def no_output_initial(self):
    self.output_initial = False
    return
  
  def output_every_iteration(self, every_it):
    """
    Define the periodic iteration at which to output the material parameter p, the gradient, and the material properties (default 0, no output).
    """
    self.output_every = every_it
    return
    
  def output_gradient(self,x=True):
    """
    Enable output the gradient of the cost function wrt p (default False)
    """
    self.output_gradient_obj = x
    return
    
  def output_gradient_constraints(self,x=True):
    """
    Enable output the constrain and their gradient wrt p (default False)
    """
    self.output_grad_constraints = x
    return
    
  def communicate_functions_names(self, cf, constrains):
    self.cf_name = cf
    self.constrains_names = constrains
  
  def output(self, it, cf, constrains_val, p_raw, grad_cf, grad_constraints, p_filtered=None, val_at=None):
    if not self.initialized: self.initiate_output() 
    
    #output to log
    out = open(self.output_log_name, 'a')
    out.write(f"{it}\t{cf:.6e}")
    for c in constrains_val:
      out.write(f"\t{c:.6e}")
    out.write("\n")
    out.close()
    
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
    dict_var["Density parameter"] = self.correct_dataset_length(p_raw, val_at)
    #add filtered density parameter p
    if p_filtered is not None:
      dict_var["Density parameter filtered"] = self.correct_dataset_length(p_filtered, val_at)
    #add gradient
    if self.output_gradient_obj:
      dict_var[f"Gradient d{self.cf_name}_dp"] = self.correct_dataset_length(grad_cf, val_at)
    #add gradient constrain
    if self.output_grad_constraints:
      for i,grad in enumerate(grad_constraints):
        dict_var[f"Gradient d{self.constrains_names[i]}_dp"] = \
                                self.correct_dataset_length(grad, val_at)
    
    self.outputter.write(dict_var, self.output_number)
    self.output_number += 1
    return
    
  def correct_dataset_length(self, X, val_at):
    #correct X_dataset if not of the size of the mesh
    #val_at 0 based
    if val_at is None: 
      return X
    if self.var_loc == "cell":
      X_new = np.zeros(self.n_elements, dtype='f8')
    else:
      X_new = np.zeros(len(self.vertices), dtype='f8')
    X_new[:] = np.nan
    X_new[val_at] = X
    return X_new
    
  def initiate_output(self):
    self.initialized = True
    #initiate log
    out = open(self.output_log_name, 'w')
    out.write(f"Iteration\t{self.cf_name}")
    for name in self.constrains_names:
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
                            self.indexes, self.var_loc)
    return
  
  def set_mesh_info(self, vertices, elements, indexes, var_loc):
    #use by the crafter to pass mesh information
    self.vertices = vertices
    self.elements = elements
    self.n_elements = 0
    for (elem_type, elems) in elements:
      self.n_elements += len(elems)
    self.indexes = indexes
    self.var_loc = var_loc
    return
    
  def write_fields_to_file(self, X, filename, Xname=None, at_ids=None):
    r"""
    Description:
      Output the field datas given in the list X using ``MeshIO`` python library.
      For a simple dataset ``x`` to write, use ``X=[x]``.
    
    Parameters:
      ``X`` (list of numpy array): The list of field datas to output
      
      ``filename`` (str): The name of the output file. Note, the format is deduced from the file extension.
      
      ``Xname`` (list of str): The dataset names. Must be ordered the same as X.
      
      ``at_ids`` (numpy array): If the X datasets does not span the whole simulation domain, the ``at_ids`` array give the point/cell ids corresponding to the given data.
      
    """
    if Xname is None:
      Xname = [f"Field{i}" for i in range(X)]
    if not isinstance(Xname, list):
      Xname = [Xname]
    for x in X:
      if at_ids is not None:
        X_ = self.correct_dataset_length(X, at_ids)
      else:
        X_ = X
    dict_var = {}
    for i,x in enumerate(X):
      data = []
      for (elem_type, index) in self.indexes:
        data.append(x[index])
      dict_var[Xname[i]] = data
    if self.var_loc == "cell":
      mesh = meshio.Mesh(self.vertices, self.elements, cell_data=dict_var)
    elif self.var_loc == "point":
      mesh = meshio.Mesh(self.vertices, self.elements, point_data=dict_var)
    mesh.write(filename)
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
    
  def write(self, dict_var, n_output):
    #transform dict_var
    for var, data in dict_var.items():
      new_data = []
      for (elem_type, index) in self.indexes:
        new_data.append(data[index])
      dict_var[var] = new_data
    if self.var_loc == "point":
      mesh = meshio.Mesh(self.vertices, self.elements, point_data=dict_var)
    elif self.var_loc == "cell":
      mesh = meshio.Mesh(self.vertices, self.elements, cell_data=dict_var)
    out = self.filename + '-' + str(n_output) + "." + self.fileformat
    mesh.write(out)
    self.n_output += 1
    return
  
  def set_mesh(self, vertices, elements, indexes, var_loc):
    self.vertices = vertices
    self.elements = elements
    self.indexes = indexes
    self.var_loc = var_loc
    return
