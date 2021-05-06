# This file is a part of the topology optimization program by Moise Rousseau

import numpy as np
import h5py

from ..PFLOTRAN import default_water_density, default_gravity, default_viscosity
from .common import __cumsum_from_connection_to_array__



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
      self.connections_to_sum = np.array(connections) 
      if len(self.connections_to_sum.shape) != 2:
        print("Something went wrong with the connections provided:")
        print(connections)
        exit(1)
      if (True in (self.connections_to_sum <= 0)):
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
    self.sign = None
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
    self.name = "Flux Sum"
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
    return X_n1 * X_n2 / (self.d_fraction * X_n1 + (1-self.d_fraction) * X_n2)
    

  
  ### COST FUNCTION ###
  def evaluate(self, p):
    """
    Evaluate the cost function
    Return the sum of the flux calculated at the given connection [m3/s]
    """
    if not self.initialized: self.__initialize__()
    k_con = self.interpole_at_face(self.k)[self.mask]
    d_P_con = (self.pressure[self.connection_ids[:,1]-1] - 
                        self.pressure[self.connection_ids[:,0]-1])[self.mask]
    eg = default_water_density * default_gravity
    cf = - self.sign * self.area_con * k_con / default_viscosity * \
                (d_P_con + eg * self.d_z_con) / self.distance_con
                
    if self.squared: 
      cf *= - self.sign * (d_P_con + eg * self.d_z_con) / self.distance_con
    return np.sum(cf)
  
  
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
      d_P_con = self.pressure[self.connection_ids[:,1][self.mask]-1] - \
                               self.pressure[self.connection_ids[:,0][self.mask]-1]
      eg = default_water_density * default_gravity
      deriv = -2 * abs(self.sign) * self.area_con * k_con / default_viscosity * \
                                   (d_P_con + eg * self.d_z_con) / self.distance_con**2
    else:
      deriv = self.sign * self.area_con * k_con / (self.distance_con * \
                                                                   default_viscosity)
    
    __cumsum_from_connection_to_array__(self.dobj_dP, 
                                     self.connection_ids[:,0][self.mask]-1,
                                     deriv)
    __cumsum_from_connection_to_array__(self.dobj_dP, 
                                     self.connection_ids[:,1][self.mask]-1,
                                     -deriv)
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
    Must have the size of the inputs
    """
    if not self.initialized: self.__initialize__()
    if self.dobj_dmat_props[2] is None:
      self.dobj_dmat_props[2] = np.zeros(len(self.pressure),dtype='f8')
    else:
      self.dobj_dmat_props[2][:] = 0.
      
    K1 = self.k[self.connection_ids[:,0][self.mask]-1] 
    K2 = self.k[self.connection_ids[:,1][self.mask]-1]
    den = ( self.d_fraction_con*K1 + (1-self.d_fraction_con)*K2 ) ** 2
    d_P_con = self.pressure[self.connection_ids[:,1][self.mask]-1] - \
                             self.pressure[self.connection_ids[:,0][self.mask]-1]
    eg = default_water_density * default_gravity
    prefactor = -self.sign * self.area_con *  (d_P_con + eg * \
                              self.d_z_con) / (self.distance_con * default_viscosity)
    if self.squared: 
      prefactor *= -self.sign * (d_P_con + eg * self.d_z_con) / self.distance_con
    dK1 = prefactor * K2**2 * (1-self.d_fraction_con) / den
    dK2 = prefactor * K1**2 * self.d_fraction_con / den

    __cumsum_from_connection_to_array__(self.dobj_dmat_props[2], \
                                     self.connection_ids[:,0][self.mask]-1, dK1)
    __cumsum_from_connection_to_array__(self.dobj_dmat_props[2], \
                                     self.connection_ids[:,1][self.mask]-1, dK2)
    return
  
  
  
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
    print(f"Current {self.name}: {cf:.6e}")
    return cf
  
  
  ### INITIALIZER FUNCTION ###
  def __initialize__(self):
    """
    Initialize the derived quantities from the inputs that must be compute one 
    time only.
    """
    self.initialized = True
    #ger id of the connections to sum
    if self.connections_to_sum is None:
      self.mask = (self.connection_ids[:,0] > 0) * (self.connection_ids[:,1] > 0)
      self.sign = 1.
    else:
      #transform a to have unique key
      offset = int(10**(np.ceil(np.log10(np.max(self.connection_ids)))+1))
      unique_id_con = offset*self.connection_ids[:,0] + self.connection_ids[:,1]
      unique_id_to_sum = offset*self.connections_to_sum[:,0] + self.connections_to_sum[:,1]
      self.mask = np.isin(unique_id_con, unique_id_to_sum)
      self.sign = np.where(self.mask, 2., 0.)
      #revert the unique id to sum key
      unique_id_to_sum = offset*self.connections_to_sum[:,1] + self.connections_to_sum[:,0]
      self.mask += np.isin(unique_id_con, unique_id_to_sum)
      self.sign += np.where(self.mask, -1., 0.)
      self.sign = self.sign[self.mask]
      test = len(self.connections_to_sum) - len(np.where(self.mask)[0])
      if test < 0:
        print("\n###\nWARNING!\nSome connections are duplicated!\n###\n")
      elif test > 0:
        print("\n###\nWARNING!\nSome connections were missed!\n###\n")
    
    #compute constant at the face of interest
    if self.area_con is None: self.area_con = self.areas[self.mask]
    if self.distance_con is None: self.distance_con = self.distance[self.mask]
    if self.d_fraction_con is None: self.d_fraction_con = self.d_fraction[self.mask]
    if self.d_z_con is None:
      self.d_z_con = self.z[self.connection_ids[:,1]-1] - self.z[self.connection_ids[:,0]-1]
      self.d_z_con = self.d_z_con[self.mask]
      
    return
    
  
  
  ### REQUIRED FOR CRAFTING ###
  def __require_adjoint__(self): return "RICHARDS"
  def __get_PFLOTRAN_output_variable_needed__(self):
    return self.output_variable_needed
  def __get_name__(self): return self.name

