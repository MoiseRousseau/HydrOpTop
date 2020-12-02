import numpy as np
import h5py
import subprocess

default_gravity = 9.8068
default_viscosity = 8.904156e-4
default_water_density = 997.16

class PFLOTRAN:
  """
  This class make the interface between PFLOTRAN and the calculation
  of sensitivity derivative and the input
  """
  def __init__(self, pft_in):
    #input related
    self.pft_in = pft_in
    self.input_folder = '/'.join(pft_in.split('/')[:-1])+'/'
    if self.input_folder[0] == '/': self.input_folder = '.' + self.input_folder
    self.__read_input_file__()
    self.__get_mesh_info__()
    self.__get_nvertices_ncells__()
    
    #running
    self.mpicommand = ""
    self.nproc = 1
    
    #output
    self.pft_out = '.'.join(pft_in.split('.')[:-1])+'.h5'
    self.dict_var_out = {"VOLUME":"Volume", "FACE_AREA": "Face Area", 
                         "LIQUID_PRESSURE":"Liquid Pressure",
                         "Z_COORDINATE":"Z Coordinate"}
    self.dict_var_in = {"PERMEABILITY":"K_Sensitivity.mat",
                        "LIQUID_PRESSURE":"Rjacobian.mat"}
    return
    
  def parallel_calling_command(self, processes, command):
    """
    Specify the command line argument for running PFLOTRAN related to 
    parallelization.
    Arguments:
    - processes: the number of core to run PFLOTRAN (the -n argument for mpirun)
    - command: the MPI command (example: mpiexec.mpich)
    """
    self.mpicommand = command
    self.nproc = processes
    return
  
  
  
  # interacting with data #
  def get_region_ids(self, reg_name):
    if self.mesh_type == "ugi" or self.mesh_type == "h5": #unstructured
      pass #TODO
    elif self.mesh_type == "uge":
      pass #TODO
    else:
      print("Unsupported mesh type in get_region_ids()")
    return ids
  
  def create_cell_indexed_dataset(self, X_dataset, dataset_name, h5_file_name="",
                                  X_ids=None, resize_to=True):
    """
    Create a PFLOTRAN cell indexed dataset.
    Arguments:
    - X_dataset: the dataset
    - dataset_name: its name (need to be the same as in PFLOTRAN input deck)
    - h5_file_name: the name of the h5 output file (same as in PFLOTRAN input deck)
    - X_ids: the cell ids matching the dataset value in X
             (i.e. if X_ids = [5, 3] and X_dataset = [1e7, 1e8],
             therefore, cell id 5 will have a X of 1e7 and 3 with 1e8).
             By default, assumed in natural order
    - resize_to: boolean for resizing the given dataset to number of cell
                 (default = True)
    """
    #first cell is at i = 0
    if not h5_file_name: h5_file_name=dataset_name.lower()+'.h5'
    out = h5py.File(h5_file_name, 'w')
    if resize_to and self.n_cells != len(X_dataset):
      X_new = np.zeros(self.n_cells, dtype='f8')
      X_new[X_ids-1] = X_dataset
      X_dataset = X_new
      X_ids = None
    out.create_dataset(dataset_name, data=X_dataset)
    if X_ids is None:
      out.create_dataset("Cell Ids", data=np.arange(1,len(X_dataset)+1))
    else:
      out.create_dataset("Cell Ids", data=X_ids)
    out.close()
    return

  
  # running
  def run_PFLOTRAN(self):
    """
    Run PFLOTRAN. No argument method
    """
    if self.mpicommand:
      cmd = [self.mpicommand, "-n", self.nproc, "pflotran", "-pflotranin", self.pft_in]
    else:
      cmd = ["pflotran", "-pflotranin", self.pft_in]
    ret = subprocess.call(cmd, stdout=open("PFLOTRAN_simulation.log",'w'))
    if ret: 
      print("\n!!! Error occured in PFLOTRAN simulation !!!\n")
      exit()
    return 
  
  
  # interact with output data
  def initiate_output_cell_variable(self):
    return np.zeros(self.n_cells, dtype='f8')
  
  
  def get_output_variable(self, var, out=None, i_timestep=-1):
    """
    Return output variable after simulation
    If out array is provided, copy variable to array, else, create a new one
    Arguments:
    - var: the variable name as in PFLOTRAN input file under VARIABLES block
    - out: the numpy output array (default=None)
    - timestep: the i-th timestep to extract
    """
    src = h5py.File(self.pft_out, 'r')
    timesteps = [x for x in src.keys() if "Time" in x]
    right_time = timesteps[i_timestep]
    key_to_find = self.dict_var_out[var]
    found = False
    for out_var in src[right_time].keys():
      if key_to_find in out_var: 
        found = True
        break
    if not found:
      print(f"\nOutput variable \"{self.dict_var_out[var]}\" not found in PFLOTRAN output")
      print(f"Have you forgot to add the \"{var}\" output variable under the OUTPUT card?\n")
      exit(1)
    if out is None:
      out = np.array(src[right_time + '/' + out_var])
    else:
      out[:] = np.array(src[right_time + '/' + out_var])
    return out
  
  def get_sensitivity(self, var):
    """
    Return a (3,n) shaped numpy array (I, J, data) representing the derivative
    of the residual according to the inputed variable. Input variable must be 
    consistent with a material property in the input deck
    Arguments:
    - var: the input variable (ex: PERMEABILITY)
    """
    f = self.input_folder + self.dict_var_in[var]
    data = np.genfromtxt(f, skip_header=8, skip_footer=2)
    return data
  
  def write_cell_variable_XDMF(var, var_name="Var", out_file="out.h5", out_xmf = "out.xmf",
	                             var_ids = None, link_to_pft_out="", default_val=np.nan):
	  #open or create output file
    try:
      out = h5py.File(out_file, 'r+')
    except:
      out = h5py.File(out_file, 'w')
	  #delete var_name dataset if already exist
    if var_name in list(out.keys()): del out[var_name]
    out.create_dataset(var_name, data=var)
    out.close()
    return
  
  def update_sensitivity(self, var, coo_mat):
    f = self.input_folder + self.dict_var_in[var]
    data = np.genfromtxt(f, skip_header=8, skip_footer=2)
    coo_mat.data[:] = data[:,2]
    return 
  
  
  
  ### PRIVATE METHOD ###
  def __read_input_file__(self):
    """
    Store input deck
    """
    input_file = self.input_folder + self.pft_in 
    src = open(input_file, 'r')
    self.input_deck = src.readlines()
    src.close()
  
  def __get_mesh_info__(self):
    """
    Read PFLOTRAN input deck to get the mesh type and the mesh file
    """
    for line in self.input_deck:
      if "TYPE" and "UNSTRUCTURED_EXPLICIT" in line: 
        self.mesh_type = "uge"
        break
      if "TYPE" and "UNSTRUCTURED" in line: 
        if ".h5" in line: self.mesh_type = "h5"
        else: self.mesh_type = "ugi"
        break
      if "TYPE" and "STRUCTURED" in line: 
        self.mesh_type = "struc"
        print("\nWARNING: STRUCTURED grid type not supported\n")
        exit()
        break
    nline = line.split()
    for i,x in enumerate(nline):
      if "STRUCTURED" in x: break
    self.mesh_file = nline[i+1]
    return
  
  def __get_nvertices_ncells__(self):
    """
    Get the number of vertices and cells in the input mesh
    """
    mesh_path = self.input_folder + self.mesh_file
    if self.mesh_type == "h5": #unstructed h5 mesh
      src = h5py.File(mesh_path, 'r')
      self.n_vertices = len(src["Domain/Vertices"])
      self.n_cells = len(src["Domain/Cells"])
      src.close()
      return
    elif self.mesh_type == "ugi": #unstructured ascii mesh
      src = open(mesh_path, 'r')
      line = src.readline()
      self.n_cells, self.n_vertices = [int(x) for x in line.split()]
      src.close()
      return
    elif self.mesh_type == "uge": #unstructured explicit
      src = open(mesh_path, 'r')
      self.n_cells = int(src.readline())
      src.close()
      self.n_vertices = -1 #no info about it...
      return
    
    
    
