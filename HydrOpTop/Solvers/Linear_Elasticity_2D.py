import numpy as np
from scipy.sparse import coo_matrix
import h5py
import subprocess
import time
import struct
import pathlib
import os



class Linear_Elasticity_2D:
  def __init__(self, prefix, poisson_ratio=0.3):
    self.prefix = prefix
    self.meshfile = prefix + ".mesh"
    self.matpropfile = prefix + ".matprops"
    
    #get problem size
    src = open(self.meshfile, 'r')
    self.n_points = int(src.readline())
    for i in range(self.n_points):
      src.readline()
    self.n_cells = int(src.readline())
    src.close()
    
    #get poisson ratio
    self.poisson_ratio = poisson_ratio
    
    self.areas = None #areas of elements
    self.element_center = None
    self.solver_command = str(pathlib.Path(__file__).parent.absolute()) + "/MinimalFEM/MinimalFEM"
    self.no_run = False
    return
    
  def get_grid_size(self):
    return self.n_cells
  
  def disable_run(self):
    """
    Disable running simulation (for debug purpose)
    """
    self.no_run = True
    return
  
  def get_var_location(self):
    return "cell" #or "point"
  
  
  def create_cell_indexed_dataset(self, X_dataset, outfile="",
                                        X_ids=None, resize_to=True):
    """
    Write a variable given by the Crafter to be read by the solver
    Arguments:
    - X_dataset: the dataset
    - X_ids: the cell ids matching the dataset value in X
             (i.e. if X_ids = [5, 3] and X_dataset = [1e7, 1e8],
             therefore, cell id 5 will have a X of 1e7 and 3 with 1e8).
             By default, assumed in natural order
    - resize_to: boolean for resizing the given dataset to number of cell
                 (default = True)
    """
    out = open(self.prefix+".matprops",'w')
    out.write(str(self.poisson_ratio) + '\n')
    for x in X_dataset:
      out.write(f"{x}\n")
    out.close()
    return
   
    
  def get_mesh(self):
    """
    Return the mesh in meshio format
    """
    vert = np.zeros((self.n_points, 3), dtype='f8')
    elems = np.zeros((self.n_cells, 3), dtype='i8')
    src = open(self.meshfile, 'r')
    src.readline()
    for i in range(self.n_points):
      vert[i,0:2] = [float(x) for x in src.readline().split()]
    src.readline()
    for i in range(self.n_cells):
      elems[i] = [int(x) for x in src.readline().split()]
    src.close()
    cells = [("triangle",elems)]
    indexes = [("triangle",np.arange(self.n_cells))]
    if self.areas is None:
      self.areas = np.zeros(self.n_cells,dtype='f8')
      for i,nodes in enumerate(elems):
        u = vert[nodes[1]] - vert[nodes[0]]
        v = vert[nodes[2]] - vert[nodes[0]]
        self.areas[i] = np.dot(u,v)/2
    if self.element_center is None:
      self.element_center = np.zeros((self.n_cells,3), dtype='f8')
      self.element_center = (vert[elems[:,0]] + vert[elems[:,1]] + vert[elems[:,2]]) / 3
    return vert, cells, indexes

  
  # running
  def run(self):
    """
    Run the solver
    """
    if self.no_run: return 0
    print("Running Solver: ",end='')
    cmd = [self.solver_command, self.prefix] 
    tstart = time.time()
    ret = subprocess.call(cmd)
    print(f"{time.time() - tstart} s to run simulation")
    if ret: 
      print("\n!!! Error occured in Solver !!!")
      print("Please see " + self.log)
    return ret
  
  
  
  def get_output_variable(self, var, out=None, i_timestep=-1):
    """
    Return output variable after simulation
    If out array is provided, copy variable to array, else, create a new one
    Arguments:
    - var: the variable name as in PFLOTRAN input file under VARIABLES block
    - out: the numpy output array (default=None)
    - timestep: the i-th timestep to extract
    """
    if var == "STRESS":
      f = self.prefix + '.stress'
      temp = np.genfromtxt(f)
    elif var == "DISPLACEMENTS":
      f = self.prefix + '.displacements'
      temp = np.genfromtxt(f)
      temp = temp.flatten()
    elif var == "VOLUME":
      if self.areas is None:
        self.get_mesh()
      temp = self.areas
    elif var == "MECHANICAL_LOAD":
      f = self.prefix + '.bcs'
      src = open(f,'r')
      count = int(src.readline())
      for i in range(count):
        src.readline()
      count = int(src.readline())
      temp = np.zeros(self.n_points*2, dtype='f8')
      for i in range(count):
        line = src.readline().split()
        node = int(line[0])
        xload = float(line[1])
        yload = float(line[2])
        temp[2*node] = xload
        temp[2*node+1] = yload
      src.close()
    elif var == "ELEMENT_CENTER_X":
      temp = self.element_center[:,0]
    elif var == "ELEMENT_CENTER_Y":
      temp = self.element_center[:,1]
    elif var == "ELEMENT_CENTER_Z":
      temp = self.element_center[:,2]
    else:
      print(f"No variable {var} with solver Linear_Elasticity_2D")
      exit(1)
    
    if out is not None:
      out[:] = temp
    else:
      out = temp
    return out
  
  def get_sensitivity(self, var, timestep=None, coo_mat=None):
    """
    Return a scipy COO matrix representing the derivative
    of the residual according to the given variable. 
    - var: the input variable to get sensitivity
    - timestep: the timestep 
    - coo_mat: a scipy COO matrix for in place assignment
    """
    if var == "DISPLACEMENTS":
      filename = self.prefix + "_jacobian.bin"
    elif var == "YOUNG_MODULUS":
      filename = self.prefix + "_sensitivity.bin"
    else:
      print(f"Unknown variable {var} sensibility")
      exit(1)
      
    f = open(filename, 'rb')
    rows = struct.unpack('q', f.read(8))[0]
    cols = struct.unpack('q', f.read(8))[0]
    nnzs = struct.unpack('q', f.read(8))[0]
    outS = struct.unpack('q', f.read(8))[0]
    innS = struct.unpack('q', f.read(8))[0]
      
    val = np.zeros(nnzs, 'f8')
    for count in range(nnzs):
      val[count] = struct.unpack('d', f.read(8))[0]
    
    if coo_mat is None:
      i = np.zeros(outS, 'i8')
      j = np.zeros(nnzs, 'i8')
      for count in range(outS):
        i[count] = struct.unpack('I', f.read(4))[0]
      for count in range(nnzs):
        j[count] = struct.unpack('I', f.read(4))[0]
      ii = np.zeros(nnzs, 'i8')
      count = 0
      for k in range(outS-1):
        ii[i[count]:i[count+1]] = count
        count += 1
      ii[i[count]:] = count
      coo_mat = coo_matrix((val, (j, ii)), shape=(rows,cols))
      return coo_mat
    else:
      coo_mat.data[:] = val
    return
    return data
  
  def __get_and_compile_solver__():
    """
    An automated method to download and compile the solver
    """
    install_folder = str(pathlib.Path(__file__).parent.absolute()) + "/MinimalFEM/"
    #download solver
    cmd = ["git", "clone", "https://github.com/MoiseRousseau/MinimalFem-For-Topology-Optimization", install_folder]
    subprocess.run(cmd)
    #compile it
    cmd = ["make"]
    subprocess.run(cmd, cwd=install_folder)
    return
    
    
