import numpy as np
from scipy.sparse import coo_matrix, dia_matrix
import h5py
import subprocess
import os
import time

from .Base_Simulator import Base_Simulator
from ..utils import geometry_utils as geom

DEFAULT_GRAVITY = 9.8068
DEFAULT_VISCOSITY = 8.904156e-4
DEFAULT_DENSITY = 997.16
DEFAULT_REF_PRESSURE = 101325.0

class PFLOTRAN(Base_Simulator):
  r"""
  Description:
    IO shield to PFLOTRAN solver
  
  Parameters:
    ``pflotranin`` (str): path to the PFLOTRAN input file
    
    ``mesh_info`` (str): path to a PFLOTRAN output file containing simulation independant PFLOTRAN output variable such as the mesh informations (face area or cell volume for example).
    Providing mesh information can help reduce the size of PFLOTRAN output file at every iteration, therefore saving time and increase the life of your SSD!
  
  Differentiated design variables: 
    ``PERMEABILITY``
  
  Differentiated output variables:
    ``PRESSURE``
  
  """
  def __init__(self, pft_in, dry_run=False, mesh_info=None):
    #input related
    self.pft_in = pft_in
    self.dry_run = dry_run
    self.input_folder = '/'.join(pft_in.split('/')[:-1])+'/'
    self.prefix = (self.pft_in.split('/')[-1]).split('.')[0]
    self.output_sensitivity_format = "HDF5" #default
    if self.input_folder[0] == '/': self.input_folder = '.' + self.input_folder
    self.__get_input_deck__(self.pft_in)
    self.mesh_type = None
    self.__get_mesh_info__()
    self.__get_nvertices_ncells__()
    self.__get_sensitivity_info__()
    
    #running
    self.mpicommand = ""
    self.nproc = 1
    self.dry_run = False #boolean flag to not run PFLOTRAN for debugging
    
    #output
    self.solved_variables = ["LIQUID_HEAD", "LIQUID_PRESSURE"]
    self.pft_out = '.'.join(pft_in.split('.')[:-1])+'.h5'
    self.pft_out_sensitivity = '.'.join(pft_in.split('.')[:-1]) + "-sensitivity-flow"
    if self.mesh_type in ["ugi", "h5"]:
      self.domain_file = self.pft_out
    else:
      self.domain_file = self.__get_domain_filename__()
    if mesh_info is None:
      self.mesh_info = self.pft_out
      self.mesh_info_present = False
    else:
      self.mesh_info = mesh_info
      self.mesh_info_present = True
    
    #for internal working
    self.dict_var_out = {
        "FACE_AREA" : "Face Area", 
        "FACE_DISTANCE_BETWEEN_CENTER" : "Face Distance Between Center",
        "FACE_UPWIND_FRACTION" : "Face Upwind Fraction",
        "FACE_NORMAL_X": "Face Normal X Component",
        "FACE_NORMAL_Y": "Face Normal Y Component",
        "FACE_NORMAL_Z": "Face Normal Z Component",
        "FACE_CELL_CENTER_VECTOR_X": "Face Cell Center X Component",
        "FACE_CELL_CENTER_VECTOR_Y": "Face Cell Vector Y Component",
        "FACE_CELL_CENTER_VECTOR_Z": "Face Cell Vector Z Component",
        "LIQUID_CONDUCTIVITY" : "Liquid Conductivity",
        "LIQUID_PRESSURE" : "Liquid Pressure",
        "LIQUID_HEAD" : "Liquid Pressure",
        "PERMEABILITY" : "Permeability",
        "VOLUME" : "Volume", 
        "ELEMENT_CENTER_X" : "X Coordinate",
        "ELEMENT_CENTER_Y" : "Y Coordinate",
        "ELEMENT_CENTER_Z" : "Z Coordinate"
    }
    self.dict_var_sensitivity_matlab = \
         {"PERMEABILITY":"permeability","LIQUID_PRESSURE":"pressure"}
    self.dict_var_sensitivity_hdf5 = {
         "PERMEABILITY":"Permeability []",
         "LIQUID_PRESSURE":"Pressure []",
         "LIQUID_HEAD":"Pressure []",
     }
    return
  
  def set_polyhedral_mesh_file(self, x):
    self.domain_file = x
    return
    
  def set_parallel_calling_command(self, processes, command):
    r"""
    Description:
      Set the number of process :math:`n` to run and the command to call (default is `mpiexec.mpich`)
    
    Parameters:
      ``processes`` (int): the number of core to run PFLOTRAN (the -n argument for mpirun)
      ``command`` (str): the MPI command (example: ``mpiexec.mpich``)
    
    """
    self.mpicommand = command
    self.nproc = processes
    return
    
  def get_grid_size(self):
    """
    Return the number of element in the mesh.
    """
    return self.n_cells
  
  def get_var_location(self, var):
    return "cell"
  
  
  # interacting with data #
  def get_region_ids(self, reg_name):
    """
    Interact with the REGION card defined in the PFLOTRAN input file
    Return the cell Ids associated with the region
    
    :param reg_name: Name of the region in PFLOTRAN input file
    :type reg_name: str
    """
    if reg_name == "__all__":
      return np.arange(0,self.n_cells)+1
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
    
    #try hdf5
    ext = filename.split('.')[-1]
    try:
      src = h5py.File(filename, 'r')
      if reg_name in src["Regions"]:
        cell_ids = np.array(src["Regions/"+reg_name+"/Cell Ids"])
        src.close()
      else:
        src.close()
        print(f"Region not found in mesh file {filename}")
        exit(1)
    #else ascii
    except:
      try:
        cell_ids = np.genfromtxt(filename, dtype='i8')
      except Exception as e:
        raise IOError(e)
    return cell_ids
  
  def get_connections_ids_integral_flux(self, integral_flux_name):
    r"""
    Description:
      Read the INTEGRAL_FLUX card and return the faces considered
    
    Parameters:
      ``integral_flux_name`` (str): The name of the integral flux
    
    Return:
      A ``numpy`` array of size (n,2) of the n faces defined by the two cell ids sharing each face.
    
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
    while "CELL_IDS" not in line:
      i += 1
      line = self.input_deck[i]
      for x in ["POLYGON", "COORDINATES_AND_DIRECTIONS", 
                "PLANE", "VERTICES", "END", "/"]:
        if x in line:
          print("Only INTEGRAL_FLUX defined with CELL_IDS are supported at this time")
          exit(1)
    cell_ids = []
    i += 1
    line = self.input_deck[i].split()
    while "/" not in line and "END" not in line:
      cell_ids.append([int(line[0]),int(line[1])])
      i += 1
      line = self.input_deck[i].split()
    if len(cell_ids) == 0:
      print(f"No connections found in the INTEGRAL_FLUX card \"{integral_flux_name}\"")
      exit(1)
    return np.array(cell_ids)
  
  def create_cell_indexed_dataset(self, X_dataset, dataset_name, h5_file_name="",
                                  X_ids=None, resize_to=True):
    r"""
    Description:
      Create a PFLOTRAN cell indexed dataset.
      
    Parameters:
      ``X_dataset`` (numpy array): The dataset to write
      
      ``dataset_name`` (str): he dataset name (need to be the same as in PFLOTRAN input deck)
      
      ``h5_file_name`` (str): the name of the h5 output file (same as in PFLOTRAN input deck)
      
      ``X_ids`` (numpy array): the cell ids matching the dataset value in X
      (i.e. if X_ids = [5, 3] and X_dataset = [1e7, 1e8], therefore, cell id 5
      will have a X of 1e7 and 3 with 1e8). By default, assumed in natural order
      
      ``resize_to`` (bool): boolean for resizing the given dataset to number of cell (default = True)
    
    """
    resize_to = True # Enforce this or PFLOTRAN will return an error
    #first cell is at i = 0
    if not h5_file_name: h5_file_name=dataset_name.lower()+'.h5'
    h5_file_name = self.input_folder + h5_file_name
    out = h5py.File(h5_file_name, 'w')
    if X_ids is None: resize_to=False
    if resize_to and self.n_cells != len(X_dataset):
      if X_ids is None:
        print("Error: user must provide the cell ids corresponding to the dataset since the length of the dataset length does not match the number of cell in the grid")
        exit(1)
      X_new = np.zeros(self.n_cells, dtype='f8')
      X_new[X_ids.astype('i8')-1] = X_dataset
      X_dataset = X_new
      X_ids = None
    out.create_dataset(dataset_name, data=X_dataset)
    if X_ids is None:
      out.create_dataset("Cell Ids", data=np.arange(1,len(X_dataset)+1))
    else:
      out.create_dataset("Cell Ids", data=X_ids)
    out.close()
    return

  def __get_mesh_center__(self):
    if self.mesh_type in ["ugi","h5"]:
      vert,elem = self.__get_unstructured_mesh__()
      center = np.array([
        geom.element_center(vert[e[1:e[0]+1]-1]) for e in elem
      ])
      return center
    mesh_path = self.input_folder + self.mesh_file
    if self.mesh_type in ["uge"]:
      src = open(mesh_path, 'r')
      center = [[float(x) for x in l.split()[1:4]] for l in src.readlines()[1:self.n_cells+1]]
      center = np.array(center)
      src.close()
    elif self.mesh_type in ["h5e"]:
      mesh = h5py.File(mesh_path, 'r')
      center = np.array(mesh["Domain/Cells/Centers"])
      mesh.close()
    return center
  
  def __get_mesh_volume__(self):
    mesh_path = self.input_folder + self.mesh_file
    if self.mesh_type in ["ugi","h5"]:
      vert,elem = self.__get_unstructured_mesh__()
      volume = np.array([
        geom.element_volume(vert[e[1:e[0]+1]-1]) for e in elem
      ])
    elif self.mesh_type in ["uge"]:
      src = open(mesh_path, 'r')
      volume = np.array(
        [float(l.split()[-1]) for l in src.readlines()[1:self.n_cells+1]]
      )
      src.close()
    elif self.mesh_type in ["h5e"]:
      src = h5py.File(mesh_path, 'r')
      volume = np.array(src["Domain/Cells/Volumes"])
      src.close()
    return volume
   
  def get_mesh(self):
    """
    TODO
    """
    #should return the mesh ready to be pass to meshio
    if self.mesh_type in ["ugi","h5"]:
      vert,elem = self.__get_unstructured_mesh__()
      #convert to meshio structure
      elem_type_code = {"tetra":4, "pyramid":5, "wedge":6, "hexahedron":8}
      cells = []
      for elem_type, elem_code in elem_type_code.items():
         cond = (elem[:,0] == elem_code)
         cells_of_type = elem[cond][:,1:elem_code+1]
         if len(cells_of_type):
             cells.append((elem_type, cells_of_type-1))
      #create a cell data to store natural id
      index = np.arange(0,len(elem))
      indexes = []
      for elem_type, elem_code in elem_type_code.items():
         cond = (elem[:,0] == elem_code)
         cells_of_type = index[cond]
         if np.sum(cond):
             indexes.append((elem_type,cells_of_type))
    else:
      mesh = h5py.File(self.domain_file,'r')
      vert = np.array(mesh["Domain/Vertices"])
      elem = np.array(mesh["Domain/Cells"])
      mesh.close()
      i = 0
      count = 0
      cells_temp = [[] for x in range(100)] #up to 100 faces
      indexes_temp = [[] for x in range(100)]
      while i < len(elem):
        x = elem[i]
        if x == 16: #polyhedron
          cell_face = []
          i += 1
          n_faces = elem[i]
          for face in range(n_faces):
            face_node = []
            i += 1
            n_nodes = elem[i]
            for node in range(n_nodes):
              i += 1
              face_node.append(elem[i])
            cell_face.append(face_node)
          cells_temp[n_faces].append(cell_face)
          indexes_temp[n_faces].append(count)
        else:
          print("TODO: add other cell type")
          return
        i += 1
        count += 1
      cells = []
      indexes = []
      for i,x in enumerate(cells_temp):
        if x:
          cells.append((f"polyhedron{i}", x))
          indexes.append((f"polyhedron{i}", indexes_temp[i]))
    return vert, cells, indexes

  
  # running
  def run(self):
    if self.dry_run: return 0
    print("Running PFLOTRAN: ",end='')
    if self.mpicommand:
      cmd = [self.mpicommand, "-n", str(self.nproc), "pflotran", "-pflotranin", self.pft_in]
    else:
      cmd = ["pflotran", "-pflotranin", self.pft_in]
    tstart = time.time()
    ret = subprocess.call(cmd, stdout=open("PFLOTRAN_simulation.log",'w'))
    print(f"{time.time() - tstart} s to run simulation")
    if ret: 
      print("\n!!! Error occured in PFLOTRAN simulation !!!")
      print(f"Please see {self.input_folder}PFLOTRAN_simulation.log for more details\n")
    return ret
  
  
  ### INTERACT WITH OUTPUT DATA ###
  
  def get_internal_connections(self, out=None):
    """
    Return the internal connection of the mesh
    """
    src = h5py.File(self.mesh_info, 'r')
    if "Domain" in list(src.keys()): prefix = "Domain/"
    else: prefix = ""
    try:
      out = np.array(src[prefix+"Connection Ids"])
    except:
      print(f"\nOutput variable \"Domain/Connection Ids\" not found in PFLOTRAN output")
      print(f"Do you forgot to add the \"PRINT_CONNECTION_IDS\" output variable under \
              the OUTPUT,SNAPSHOT_FILE card?\n")
      exit(1)
    src.close()
    return out


  def get_output_variables(self, vars_out, i_timestep=-1):
        """
        Return output variables after simulation (all variable in one call)
        """
        # Initialize output if None
        for var, out in vars_out.items():
            if out is None:
                out = np.zeros(self.n_cells)
                vars_out[var] = out
        for var, out in vars_out.items():
            self.get_output_variable(var, out, i_timestep)
        return vars_out


  def get_output_variable(self, var, out=None, i_timestep=-1):
    r"""
    Description:
      Return output variable after simulation.
      
    Parameters:
      ``var`` (str): the variable name as in PFLOTRAN input file under VARIABLES block
      
      ``out`` (numpy array): the numpy output array. If no array if provided, a new one is created
      
      ``timestep`` (int): the i-th timestep to extract (not working yet)
    
    """
    # Look for mesh related variable
    if "ELEMENT_CENTER_" in var:
      center = self.__get_mesh_center__()
      dict_index = {"X":0,"Y":1,"Z":2}
      out[:] = center[:,dict_index[var[-1]]]
      return out
    elif "VOLUME" in var:
      out[:] = self.__get_mesh_volume__()
      return out
    elif var == "CONNECTION_IDS":
      return self.get_internal_connections()

    # Look for simulation variable
    # Correct output variable if not present
    if var in ["FACE_NORMAL_X", "FACE_NORMAL_Y", "FACE_NORMAL_Z"] and \
              self.mesh_type in ["uge","h5e"]:
      print(f"{var} information not available, switch to FACE_CELL_CENTER_VECTOR instead")
      var = "FACE_CELL_CENTER_VECTOR_" + var[-1]
    #treat separately grid output since they could be in the mesh_info file
    if var in ["LIQUID_CONDUCTIVITY","LIQUID_PRESSURE","LIQUID_HEAD","PERMEABILITY"]:
      f_src = self.pft_out
    else:
      f_src = self.mesh_info
    # Get the variable
    src = h5py.File(f_src, 'r')
    if var == "LIQUID_HEAD": 
        right_time, out_var = self.__check_output_variable_present__(
            src, i_timestep, "LIQUID_PRESSURE"
        )
        eg = 1 / (DEFAULT_GRAVITY * DEFAULT_DENSITY)
        temp = (np.array(src[right_time + '/' + out_var]) - DEFAULT_REF_PRESSURE) * eg
        right_time, out_var = self.__check_output_variable_present__(
            src, i_timestep, "ELEMENT_CENTER_Z"
        )
        temp += np.array(src[right_time + '/' + out_var])
    else:
      right_time, out_var = self.__check_output_variable_present__(src, i_timestep, var)
      temp = np.array(src[right_time + '/' + out_var])
    if out is None:
      out = temp
    else:
      out[:] = temp
    src.close()
    return out
  
  def get_sensitivity(self, var, timestep=None, coo_mat=None):
    if self.output_sensitivity_format == "HDF5":
      f = self.pft_out_sensitivity + '.h5'
      src = h5py.File(f, 'r')
      i = np.array(src["Mat Structure/Row Indices"])
      j = np.array(src["Mat Structure/Column Indices"])
      if timestep is None: timestep = -1
      list_timestep = list(src.keys())
      temp_str = list_timestep[timestep] + '/' + self.dict_var_sensitivity_hdf5[var]
      data = np.array(src[ temp_str ])
      if var == "LIQUID_HEAD":
        eg = (DEFAULT_GRAVITY * DEFAULT_DENSITY) #TODO non constant density
        data *= eg
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
      restart = False
      for i,line in enumerate(self.input_deck):
        if "EXTERNAL_FILE" in line:
          self.input_deck.pop(i)
          line_split = line.split()
          index = line_split.index("EXTERNAL_FILE")
          self.__read_input_file__(self.input_folder+line_split[index+1],append_at_pos=i)
          restart = True
          break
      if not restart: finish = True
    return
  
  def __read_input_file__(self, filename, append_at_pos=0):
    """
    Store input deck
    """
    src = open(filename, 'r')
    temp = []
    #read line in source file and remove \n and commentaru
    for line in src.readlines():
      line = line.split('#')[0] 
      if len(line)>0 and line[-1] == '\n': line = line[:-1]
      if line: temp.append(line)
    #remove skip / noskip part
    skip = []
    noskip = [] 
    for i,line in enumerate(temp):
      if "NOSKIP" in line:
        noskip.append(i)
        continue
      if "SKIP" in line: 
        skip.append(i)
    if len(skip) != len(noskip):
      print(skip, noskip)
      print(f"ERROR! number of SKIP does not match the number of NOSKIP in file {filename}")
    for i,j in zip(skip, noskip):
      temp = temp[:i] + temp[j+1:]
    #add the result in the input deck
    if not self.input_deck: self.input_deck = temp
    else:
      self.input_deck = self.input_deck[:append_at_pos] + \
                        temp + self.input_deck[append_at_pos:]
    src.close()
    return 
    
  
  def __get_unstructured_mesh__(self):
      mesh_path = self.input_folder + self.mesh_file
      if self.mesh_type == "h5":
        mesh = h5py.File(mesh_path, 'r')
        vert = np.array(mesh["Domain/Vertices"])
        elem = np.array(mesh["Domain/Cells"])
        mesh.close()
      elif self.mesh_type == "ugi":
        mesh = open(mesh_path,'r')
        n_e, n_v = [int(x) for x in mesh.readline().split()]
        elem = np.zeros((n_e,9),dtype='i8')
        for iline in range(n_e):
          line = [int(x) for x in mesh.readline().split()[1:]]
          elem[iline,0] = len(line)
          elem[iline,1:len(line)+1] = line
        vert = np.zeros((n_v,3),dtype='f8')
        for iline in range(n_v):
          vert[iline] = [float(x) for x in mesh.readline().split()]
        mesh.close()
      return vert,elem

  def __get_mesh_info__(self):
    """
    Read PFLOTRAN input deck to get the mesh type and the mesh file
    """
    for line in self.input_deck:
      if "TYPE" and "UNSTRUCTURED_EXPLICIT" in line: 
        if ".h5" in line: self.mesh_type = "h5e"
        else: self.mesh_type = "uge"
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
    elif self.mesh_type == "h5e": #unstructured explicit hdf5
      src = h5py.File(mesh_path, 'r')
      self.n_cells = len(src["Domain/Cells/Volumes"])
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
  
  def __get_domain_filename__(self):
    filename = ""
    for i,line in enumerate(self.input_deck):
      if "DOMAIN_FILENAME" in line:
        line = line.split()
        index = line.index("DOMAIN_FILENAME")
        filename = line[index+1]
        break
    return filename
  
  def __check_output_variable_present__(self, src, i_timestep, var):
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
      print(f"Available variable are:")
      print(src[right_time].keys())
      print(f"Do you forgot to add the \"{var}\" output variable under the OUTPUT card?\n")
      exit(1)
    return right_time, out_var    
    
