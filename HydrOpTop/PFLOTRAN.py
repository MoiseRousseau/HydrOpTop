import numpy as np
from scipy.sparse import coo_matrix, dia_matrix
import h5py
import subprocess
import os

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
    self.prefix = (self.pft_in.split('/')[-1]).split('.')[0]
    self.output_sensitivity_format = "HDF5" #default
    if self.input_folder[0] == '/': self.input_folder = '.' + self.input_folder
    self.__get_input_deck__(self.pft_in )
    self.__get_mesh_info__()
    self.__get_nvertices_ncells__()
    self.__get_sensitivity_info__()
    
    #running
    self.mpicommand = ""
    self.nproc = 1
    
    #output
    self.pft_out = '.'.join(pft_in.split('.')[:-1])+'.h5'
    self.pft_out_sensitivity = '.'.join(pft_in.split('.')[:-1]) + "-sensitivity-flow"
    self.dict_var_out = {"FACE_AREA" : "Face Area", 
                         "FACE_DISTANCE_BETWEEN_CENTER" : "",
                         "FACE_UPWIND_FRACTION" : "",
                         "LIQUID_CONDUCTIVITY" : "Liquid Conductivity",
                         "LIQUID_PRESSURE" : "Liquid Pressure",
                         "VOLUME" : "Volume", 
                         "Z_COORDINATE" : "Z Coordinate"}
    self.dict_var_sensitivity_matlab = \
         {"PERMEABILITY":"permeability","LIQUID_PRESSURE":"pressure"}
    self.dict_var_sensitivity_hdf5 = \
         {"PERMEABILITY":"Permeability []","LIQUID_PRESSURE":"Pressure []"}
    return
    
  def set_parallel_calling_command(self, processes, command):
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
    
  def get_grid_size(self):
    return self.n_cells
  
  
  
  # interacting with data #
  def get_region_ids(self, reg_name):
    """
    Return the cell ids associated to the given region:
    - reg_name: the name of the region to get the ids.
    """
    #look for region in pflotran input
    filename = ""
    for i,line in enumerate(self.input_deck):
      if "REGION" in line and reg_name in line:
        line = self.input_deck[i+1]
        if "FILE" in line:
          line = line.split()
          index = line.index("FILE")
          filename = self.input_folder+line[index+1]
          break
    if not filename:
      print(f"No region \"{reg_name}\" found in PFLOTRAN input file, stop...")
      exit(1)
    
    if self.mesh_type == "ugi" or self.mesh_type == "uge":
      cell_ids = np.genfromtxt(filename, dtype='i8')
      
    elif self.mesh_type == "h5": #h5 region have same name in pflotran
      src = h5py.File(filename, 'r')
      if reg_name in src["Regions"]:
        cell_ids = np.array(src["Regions/"+reg_name+"/Cell Ids"])
        src.close()
      else:
        src.close()
        print(f"Region not found in mesh file {filename}")
        exit(1)
    return cell_ids
  
  def get_connections_ids_integral_flux(self, integral_flux_name):
    """
    Return the cell ids associated to the given integral flux
    Argument:
    - integral_flux_name: the name of the integral flux to get the ids.
    """
    found = False
    #find the integral flux in input deck
    for i,line in enumerate(self.input_deck):
      if "INTEGRAL_FLUX" in line:
        if integral_flux_name in line: 
          found = True
          break
    if not found:
      print(f"Integral flux {integral_flux_name} not found in input deck")
      print("Please provide the name directly after the INTEGRAL_FLUX opening card")
      exit(1)
    #check if defined by cell ids
    while "CELL_IDS" in line:
      i += 1
      line = self.input_deck[i]
      for x in ["POLYGON", "COORDINATES_AND_DIRECTIONS", 
                "PLANE", "VERTICES", "END", "/"]:
        if x in line:
          print("Only INTEGRAL_FLUX defined with CELL_IDS are supported at this time")
          exit(1)
    cell_ids = []
    while "/" in line or "END" in line:
      i += 1
      line = self.input_deck[i].split()
      cell_ids.append([line[0],line[1]])
    return cell_ids
  
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
    if X_ids is None: resize_to=False
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
      cmd = [self.mpicommand, "-n", str(self.nproc), "pflotran", "-pflotranin", self.pft_in]
    else:
      cmd = ["pflotran", "-pflotranin", self.pft_in]
    ret = subprocess.call(cmd, stdout=open("PFLOTRAN_simulation.log",'w'))
    if ret: 
      print("\n!!! Error occured in PFLOTRAN simulation !!!")
      print(f"Please see {self.input_folder}PFLOTRAN_simulation.log for more details\n")
    return ret
  
  
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
    #treat coordinate separately as they are in Domain/XC
    if var in ["XC", "YC", "ZC"]:
      if out is None:
        out = np.array(src["Domain/"+var])
      else:
        out[:] = np.array(src["Domain/"+var])
      src.close()
      return out
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
      print(f"Do you forgot to add the \"{var}\" output variable under the OUTPUT card?\n")
      exit(1)
    if out is None:
      out = np.array(src[right_time + '/' + out_var])
    else:
      out[:] = np.array(src[right_time + '/' + out_var])
    src.close()
    return out
  
  def get_sensitivity(self, var, timestep=None, coo_mat=None):
    # TODO: change the name of the dict_var_sensitivity to match the new output
    """
    Return a (3,n) shaped numpy array (I, J, data) representing the derivative
    of the residual according to the inputed variable. Input variable must be 
    consistent with a material property in the input deck.
    Sensitivity outputed by PFLOTRAN is supposed to be in matlab format
    Arguments:
    - var: the input variable (ex: PERMEABILITY)
    """
    if self.output_sensitivity_format == "HDF5":
       f = self.pft_out_sensitivity + '.h5'
       src = h5py.File(f, 'r')
       i = np.array(src["Mat Structure/Row Indices"])
       j = np.array(src["Mat Structure/Column Indices"])
       if timestep is None: timestep = -1
       list_timestep = list(src.keys())
       temp_str = list_timestep[timestep] + '/' + self.dict_var_sensitivity_hdf5[var]
       data = np.array(src[ temp_str ])
       src.close()
    elif self.output_sensitivity_format == "MATLAB":
      if timestep is None:
        output_file = [x[:-4] for x in os.listdir(self.input_folder) if x[-4:] == '.mat']
        output_file = [x for x in output_file if self.prefix+'-sensitivity-flow-' in x]
        output_file = [int(x.split('-')[-1]) for x in output_file]
        timestep = max(output_file)
      if timestep < 10: timestep = "00"+str(timestep)
      elif timestep < 100: timestep = "0"+str(timestep)
      else: timestep = str(timestep)
      f = self.pft_out_sensitivity + '-' + self.dict_var_sensitivity_matlab[var] \
            + '-' + timestep + '.mat'
      src = np.genfromtxt(f, skip_header=8, skip_footer=2)
      i, j, data = src[:,0], src[:,1], src[:,2]
    if coo_mat is None:
      new_mat = coo_matrix( (data,(i.astype('i8')-1,j.astype('i8')-1)), dtype='f8')
      return new_mat
    else:
      coo_mat.data[:] = data
    return
  
  
  
  
  ### PRIVATE METHOD ###
  def __get_input_deck__(self, filename):
    self.input_deck = []
    self.__read_input_file__(filename)
    finish = False
    while not finish:
      for i,line in enumerate(self.input_deck):
        if "EXTERNAL_FILE" in line:
          line = line.split()
          index = line.index("EXTERNAL_FILE")
          self.__read_input_file__(line[index+1])
          break
      finish = True
    return
  
  def __read_input_file__(self, filename, append_at_pos=0):
    """
    Store input deck
    """
    src = open(filename, 'r')
    temp = []
    for line in src.readlines():
      line = line.split('#')[0][:-1] #remove commentary and \n
      if line: temp.append(line) 
    if not self.input_deck: self.input_deck = temp
    else:
      self.input_deck = self.input_deck[:append_at_pos] + \
                        temp + self.input_deck[append_at_pos:]
    src.close()
    return 
    
  
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
      self.n_cells = int(src.readline().split()[1])
      src.close()
      self.n_vertices = -1 #no info about it...
      return
  
  def __get_sensitivity_info__(self):
    for i,line in enumerate(self.input_deck):
      if "SENSITIVITY_OUTPUT_FORMAT" in line:
        line = line.split()
        index = line.index("SENSITIVITY_OUTPUT_FORMAT")
        self.output_sensitivity_format = line[index+1].upper()
        break
    return
    
    
    
