# This file is a part of the topology optimization program by Moise Rousseau

import numpy as np
import h5py

from ..PFLOTRAN import default_water_density, default_gravity, default_viscosity


class Sum_Flux:
  """
  Calculate the total flux through a surface (defined by its faces) considering 
  the viscosity and density constant (for simplicity).
  Note: calculation valid only for internal face
  """
  def __init__(self, connections=None, square = False):
    """
    Connections is a 2D array storing the cell ids shared by the face
    Squared permit to sum all flux in absolute value and differentiable in 0
    """
    #objective argument
    if connections is None:
      self.connections_to_sum = None #consider all the connections
    else:
      #sort them and -1 to convert from PFLOTRAN to python indexing
      self.connections_to_sum = np.array(connections) - 1
      if len(self.connections_to_sum.shape) != 2:
        print("Something went wrong with the connections provided:")
        print(connections)
        exit(1)
      if (True in self.connections_to_sum < 0):
        print("Error: some connections implied a cell id <= 0")
        exit(1)
    self.squared = square
    #inputs
    self.d_fraction = None
    self.K = None
    self.pressure = None
    self.distance = None
    self.areas = None
    self.connection_ids = None
    #derived inputs
    self.mask = None #correspondance between connection in PFLOTRAN order and those to sum
    self.area_con = None #connection area for connection to sum
    self.d_z_con = None #d_z for connection to sum
    self.distance_con = None
    self.d_fraction_con = None
    #required attribute
    self.p_ids = None #correspondance between index in p and PFLOTRAN cell ids
                      #set by the crafter
    self.ids_p = None
    self.dobj_dp_partial = None
    self.dobj_dP = None 
    #below: derivative are 0. except for permeability. So we put None such as the function
    #d_objective_d_inputs could initialized it
    self.dobj_dmat_props = [0., 0., None, 0., 0., 0., 0., 0.] 
    self.adjoint = None #attribute storing adjoint
    self.filter = None #store the filter object
    self.initialized = False
    
    #required for problem crafting
    self.output_variable_needed = ["LIQUID_PRESSURE", "FACE_AREA",
                                   "PERMEABILITY", "FACE_UPWIND_FRACTION", 
                                   "FACE_DISTANCE_BETWEEN_CENTER",
                                   "Z_COORDINATE", "CONNECTION_IDS"] 
    return
    
  def set_ids_to_consider(self, x):
    self.ids_to_consider = x-1
    return
    
  def set_inputs(self, inputs):
    """
    Method required by the problem crafter to pass the pflotran output
    variables to the objective (the Yi)
    Inputs argument have the same size than __get_PFLOTRAN_output_variable_needed__
    Note that the inputs will be passed one time only, and will after be changed
    in-place, so that function will never be called again...
    """
    #parse input at each iteration
    self.pressure = inputs[0]
    self.areas = inputs[1]
    self.k = inputs[2]
    self.d_fraction = inputs[3]
    self.distance = inputs[4]
    self.z = inputs[5]
    self.connection_ids = inputs[6]
    return
  
  def get_inputs(self):
    return [self.pressure, self.areas, self.k, self.d_fraction, self.distance,
            self.z, self.connection_ids]
  
  def set_p_to_cell_ids(self,p_ids):
    """
    Method that pass the correspondance between the index in p and the
    PFLOTRAN cell ids.
    Required by the Crafter
    """
    self.p_ids = p_ids
    #here we need the reverse, i.e. we know the cell id in PFLOTRAN 
    #and we need the index in p
    self.ids_p = {}
    for i,x in enumerate(self.p_ids): self.ids_p[x] = i #conversion from PFLOTRAN to zeros based
    return 
    
  def set_filter(self, filter):
    self.filter = filter
    return
  
  def set_adjoint_problem(self, x):
    """
    Method used by the Crafter class to set the adjoint if needed
    """
    self.adjoint = x
    return
    
  def interpole_at_face(self, X):
    """
    Interpole the given cell centered variable at face
    Compute only the value at the connection in self.connections
    """
    X_n1 = X[self.connection_ids[:,0]-1]
    X_n2 = X[self.connection_ids[:,1]-1]
    return X_n1 * X_n2 / (self.d_fraction * X_n2 + (1-self.d_fraction) * X_n1)
    

  
  ### COST FUNCTION ###
  def evaluate_array(self, p, square):
    if not self.initialized: self.__initialize__()
    k_con = self.interpole_at_face(self.k)[self.mask]
    if not self.initialized: self.__initialize__()
    d_P_con = self.pressure[self.connections_to_sum[:,0]] - \
                             self.pressure[self.connections_to_sum[:,1]]
    cf = self.area_con * k_con / default_viscosity * \
         (d_P_con - default_water_density * default_gravity * self.d_z_con) / self.distance_con
    if square: 
      cf *= (d_P_con - default_water_density * default_gravity * self.d_z_con) / \
                                   self.distance_con
    return cf
    
  
  def evaluate(self, p):
    """
    Evaluate the cost function
    Return the sum of the flux calculated at the given connection [m3/s]
    """
    cf = self.evaluate_array(p,self.squared)
    return np.sum(cf)
  
  
  ### PARTIAL DERIVATIVES ###
  def d_objective_dP(self,p): 
    """
    Evaluate the derivative of the cost function according to the pressure.
    Required the solve the adjoint, thus must have the size of the grid
    """
    if not self.initialized: self.__initialize__()
    k_con = self.interpole_at_face(self.k)[self.mask]
    if self.dobj_dP is None:
      self.dobj_dP = np.zeros(len(self.pressure), dtype='f8') #pressure size of the grid
    else:
      self.dobj_dP[:] = 0
    if self.squared:
      deriv = 2 * self.evaluate_array(p,False) / self.distance_con
    else:
      deriv = self.area_con * k_con / (self.distance_con * default_viscosity) 
    for icon,ids in enumerate(self.connections_to_sum):
      i,j = ids
      if (i < 0) or (j < 0): continue
      if self.ids_p is not None:
        try: i = self.ids_p[i+1] #i,j 0 based, but ids_p pflotran base (1)
        except: pass
        try: j = self.ids_p[j+1]
        except: pass
      self.dobj_dP[i] += deriv[icon]
      self.dobj_dP[j] -= deriv[icon]
    return
  
  
  def d_objective_dp_partial(self,p):
    """
    Derivative of the objective function according to the density
    parameter p
    Argument:
    - p : the material parameter
    Return:
    - the derivative
    Note, could return a dummy value if the objective input does not 
    depend on p explicitely 
    Must have the size of p
    """
    if not self.initialized: self.__initialize__()
    self.dobj_dp_partial = 0.
    return None
  
  
  def d_objective_d_mat_props(self,p):
    """
    Derivative of the objective function according to other input variable
    Argument:
    - p : the material parameter
    Return:
    - A list of the derivative in the same order they are listed in 
      self.mat_props_dependance
    Note, could return a dummy value if the objective input does not 
    depend on the material properties explicitely or implicitely
    Must have the size of p
    """
    if not self.initialized: self.__initialize__()
    deriv = np.zeros(len(p),dtype='f8')
    K1 = self.k[self.connections_to_sum[:,0]] 
    K2 = self.k[self.connections_to_sum[:,1]]
    den = (self.d_fraction_con*K1 + (1-self.d_fraction_con)*K2)**2
    d_P_con = self.pressure[self.connections_to_sum[:,0]] - \
                             self.pressure[self.connections_to_sum[:,1]]
    prefactor = self.area_con *  (d_P_con - default_water_density * \
                   default_gravity * self.d_z_con) / (self.distance_con * default_viscosity)
    if self.squared: 
      prefactor *= (d_P_con - default_water_density * \
                            default_gravity * self.d_z_con) / self.distance_con
    dK1 = prefactor * K2**2 * (1-self.d_fraction_con) / den 
    dK2 = prefactor * K1**2 * self.d_fraction_con / den
    for icon,ids in enumerate(self.connections_to_sum):
      i,j = ids
      if self.ids_p is not None:
        try: i = self.ids_p[i+1]
        except: i = -1 #not in the optimized domain
        try: j = self.ids_p[j+1]
        except: j = -1
      if (i >= 0): deriv[i] += dK1[icon]
      if (j >= 0): deriv[j] += dK2[icon]
    self.dobj_dmat_props[2] = deriv
    return None
  
  
  
  ### TOTAL DERIVATIVE ###
  def d_objective_dp_total(self, p, out=None): 
    """
    Evaluate the derivative of the cost function according to the density
    parameter p. If a numpy array is provided, derivative will be copied 
    in this array, else create a new numpy array.
    """
    if not self.initialized: self.__initialize__()
    #this method could be used as is
    if out is None:
      out = np.zeros(len(self.p), dtype='f8')
    self.d_objective_dP(p) #update objective derivative wrt pressure
    self.d_objective_d_mat_props(p) #update objective derivative wrt mat prop
    self.d_objective_dp_partial(p)
    out[:] = self.adjoint.compute_sensitivity(p, self.dobj_dP, 
                 self.dobj_dmat_props, self.output_variable_needed) + self.dobj_dp_partial
    return out
  
  
  ### WRAPPER FOR NLOPT ###
  def nlopt_optimize(self,p,grad):
    """
    Wrapper to evaluate and compute the derivative of the cost function
    for calling in nlopt
    """
    #could be used as is
    cf = self.evaluate(p)
    if grad.size > 0:
      self.d_objective_dp_total(p,grad)
    print(f"Current flux: {cf:.6e}")
    return cf
  
  
  ### INITIALIZER FUNCTION ###
  def __initialize__(self):
    """
    Initialize the derived quantities from the inputs that must be compute one 
    time only.
    """
    print("Initializing Sum_Flux function...")
    self.initialized = True
    if self.connections_to_sum is None:
      no_bc_connections = (self.connection_ids[:,0] > 0) * (self.connection_ids[:,1] > 0)
      self.connections_to_sum = self.connection_ids[no_bc_connections] - 1
      self.mask = np.arange(0,len(self.connections_to_sum))
    #compute derived quantities
    #extracted quantities
    if self.mask is None:
      self.mask = np.zeros(len(self.connections_to_sum),dtype='i8')-1
      count = -1
      for i,j in self.connections_to_sum:
        count += 1
        found = False
        for k,ids in enumerate(self.connection_ids):
          if (ids[0] == i+1 and ids[1] == j+1) or \
             (ids[1] == i+1 and ids[0] == j+1):
            self.mask[count] = k
            found = True
            break
        if not found:
          print(f"Warning! Connection {i+1,j+1} missed !")
          print("This could be a non-existing connection")
    if self.area_con is None: self.area_con = self.areas[self.mask]
    if self.distance_con is None: self.distance_con = self.distance[self.mask]
    if self.d_fraction_con is None: self.d_fraction_con = self.d_fraction[self.mask]
    #derived quantities
    if self.d_z_con is None:
      self.d_z_con = self.z[self.connections_to_sum[:,0]] - \
                     self.z[self.connections_to_sum[:,1]] #only at the considered connections
    print("Done!")
    return
    
  
  
  ### REQUIRED FOR CRAFTING ###
  def __require_adjoint__(self): return "RICHARDS"
  def __get_PFLOTRAN_output_variable_needed__(self):
    return self.output_variable_needed
#  def __depend_of_mat_props__(self, var=None):
#    if var is None: return self.mat_props_dependance
#    if var in self.mat_props_dependance: return True
#    else: return False

                      
