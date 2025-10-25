# This file is a part of the topology optimization program by Moise Rousseau

import numpy as np
import h5py

DEFAULT_GRAVITY = 9.8068
DEFAULT_VISCOSITY = 8.904156e-4
DEFAULT_DENSITY = 997.16

from .common import __cumsum_from_connection_to_array__, \
                    smooth_abs_function, d_smooth_abs_function
from .Base_Function_class import Base_Function


class Sum_Flux(Base_Function):
  r"""
  Compute the flux through a given surface defined by a list of faces as:
  
  

  .. math::
      :label: sum_flux
       
       f = \sum_{(i,j) \in S} \left[A_{ij} \frac{k_{ij}}{\mu} \frac{P_i - P_j + \rho g (z_i - z_j)} {d_{ij}}\right]^n

  Faces are   specified by a the two cell ids sharing the face. 
  Fluid is considered incompressible   and with a constant viscosity (i.e. :math:`\rho` and :math:`\mu` are constant). 
  Not tested for variably saturated flow.
  
  Require PFLOTRAN outputs ``LIQUID_PRESSURE``, ``FACE_AREA``, 
  ``PERMEABILITY``, ``FACE_UPWIND_FRACTION``,
  ``FACE_DISTANCE_BETWEEN_CENTER``, ``Z_COORDINATE`` and ``CONNECTION_IDS``.

  :params connections: a two dimension array of size (N,2) storing the cell ids shared the faces on which to sum the flux. 
  :type connections: iterable
  :params option:  either one of ``"absolute"`` (each face flux are summed in absolute value),
      ``"signed"`` (each face flux are summed from cell `i` to cell `j`), or
      ``"signed_reverse"`` (each face flux are summed from cell `j` to cell `i`).
  :type option: str    
  """
  def __init__(self, connections=None, option="signed"):#, square = False):
    """
    Connections is a 2D array storing the cell ids shared by the face
    Squared permit to sum all flux in absolute value and differentiable in 0
    """
    super(Sum_Flux, self).__init__()

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
    #self.squared = square
    self.option = option
    
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
    self.initialized = False
    
    #required for problem crafting
    self.variables_needed = [
      "LIQUID_PRESSURE",
      "FACE_AREA",
      "PERMEABILITY", "FACE_UPWIND_FRACTION",
      "FACE_DISTANCE_BETWEEN_CENTER",
      "ELEMENT_CENTER_Z", "CONNECTION_IDS"
    ]
    self.name = "Flux Sum"
    return
    
  def set_ids_to_consider(self, x):
    self.ids_to_consider = x-1
    return
    
  def set_inputs(self, inputs):
    #parse input at each iteration
    self.pressure = inputs["LIQUID_PRESSURE"]
    self.areas = inputs["FACE_AREA"]
    self.k = inputs["PERMEABILITY"]
    self.d_fraction = inputs["FACE_UPWIND_FRACTION"]
    self.distance = inputs["FACE_DISTANCE_BETWEEN_CENTER"]
    self.z = inputs["ELEMENT_CENTER_Z"]
    self.connection_ids = inputs["CONNECTION_IDS"]
    return
    
  def interpole_at_face(self, X):
    """
    Interpole the given cell centered variable at face
    Compute only the value at the connection in self.connections
    """
    X_n1 = X[self.connection_ids[:,0]-1]
    X_n2 = X[self.connection_ids[:,1]-1]
    return X_n1 * X_n2 / (self.d_fraction * X_n1 + (1-self.d_fraction) * X_n2)
    
  
  def calculate_flux_at_faces(self):
    k_con = self.interpole_at_face(self.k)[self.mask]
    d_P_con = (self.pressure[self.connection_ids[:,1]-1] - 
                        self.pressure[self.connection_ids[:,0]-1])[self.mask]
    eg = DEFAULT_DENSITY * DEFAULT_GRAVITY
    fluxes = - self.sign * self.area_con * k_con / DEFAULT_VISCOSITY * \
                (d_P_con + eg * self.d_z_con) / self.distance_con
    return fluxes
  
  
  ### COST FUNCTION ###
  def evaluate(self, p):
    """
    Evaluate the cost function
    Return the sum of the flux calculated at the given connection [m3/s]
    """
    if not self.initialized: self.__initialize__()
    fluxes = self.calculate_flux_at_faces()
                
    if self.option == "signed_reverse":
      fluxes = -fluxes
    elif self.option == "absolute":
      fluxes = smooth_abs_function(fluxes)
    
    #if self.squared: 
    #  cf *= - self.sign * (d_P_con + eg * self.d_z_con) / self.distance_con
    return np.sum(fluxes)

  
  def d_objective(self, var, p):
    """
    Given a variable, return the derivative of the cost function according to that variable.
    Use current simulation state
    """
    if not self.initialized: self.__initialize__()
    if var == "LIQUID_HEAD":
      raise NotImplementedError()
    elif var == "LIQUID_PRESSURE":
      k_con = self.interpole_at_face(self.k)[self.mask]
      eg = DEFAULT_DENSITY * DEFAULT_GRAVITY
      deriv = self.sign * self.area_con * k_con / (self.distance_con * \
                                                                     DEFAULT_VISCOSITY)
      if self.option == "signed_reverse":
        deriv = -deriv
      elif self.option == "absolute":
        fluxes = self.calculate_flux_at_faces()
        deriv = d_smooth_abs_function(fluxes) * deriv
      
      #if self.squared:
      #  d_P_con = self.pressure[self.connection_ids[:,1][self.mask]-1] - \
      #                           self.pressure[self.connection_ids[:,0][self.mask]-1]
      #  eg = default_water_density * default_gravity
      #  deriv = -2 * abs(self.sign) * self.area_con * k_con / default_viscosity * \
      #                               (d_P_con + eg * self.d_z_con) / self.distance_con**2
      # Accumulator
      dobj = np.zeros(len(self.pressure), dtype='f8')
      __cumsum_from_connection_to_array__(
        dobj, self.connection_ids[:,0][self.mask]-1, deriv
      )
      __cumsum_from_connection_to_array__(
        dobj, self.connection_ids[:,1][self.mask]-1, -deriv
      )
    elif var == "PERMEABILITY":
      dobj = np.zeros(len(self.k), dtype='f8')
      K1 = self.k[self.connection_ids[:,0][self.mask]-1]
      K2 = self.k[self.connection_ids[:,1][self.mask]-1]
      den = ( self.d_fraction_con*K1 + (1-self.d_fraction_con)*K2 ) ** 2
      d_P_con = self.pressure[self.connection_ids[:,1][self.mask]-1] - \
                               self.pressure[self.connection_ids[:,0][self.mask]-1]
      eg = DEFAULT_DENSITY * DEFAULT_GRAVITY
      prefactor = -self.sign * self.area_con *  (d_P_con + eg * \
                                self.d_z_con) / (self.distance_con * DEFAULT_VISCOSITY)
      #if self.squared:
      #  prefactor *= -self.sign * (d_P_con + eg * self.d_z_con) / self.distance_con
      dK1 = prefactor * K2**2 * (1-self.d_fraction_con) / den
      dK2 = prefactor * K1**2 * self.d_fraction_con / den

      if self.option == "signed_reverse":
        dK1 = -dK1
        dK2 = -dK2
      elif self.option == "absolute":
        fluxes = self.calculate_flux_at_faces()
        dK1 = dK1 * d_smooth_abs_function(fluxes)
        dK2 = dK2 * d_smooth_abs_function(fluxes)

      __cumsum_from_connection_to_array__(
        dobj,
        self.connection_ids[:,0][self.mask]-1,
        dK1
      )
      __cumsum_from_connection_to_array__(
        dobj,
        self.connection_ids[:,1][self.mask]-1,
        dK2
      )
    else:
      print(f"No derivation for variable {var}")
      dobj = 0.
    return dobj

  
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
      unique_id_con = offset*(self.connection_ids[:,0]).astype('i8') + self.connection_ids[:,1]
      unique_id_to_sum = offset*self.connections_to_sum[:,0] + self.connections_to_sum[:,1]
      self.mask = np.isin(unique_id_con, unique_id_to_sum)
      self.sign = np.where(self.mask, 2., 0.)
      #revert the unique id to sum key
      unique_id_to_sum = offset*self.connections_to_sum[:,1] + self.connections_to_sum[:,0]
      self.mask += np.isin(unique_id_con, unique_id_to_sum)
      self.sign += np.where(self.mask, -1., 0.)
      self.sign = self.sign[self.mask]
      test = len(self.connections_to_sum) - len(np.where(self.mask)[0])
      if len(self.sign) < 1:
        print("ERROR! No connections found!")
        exit(1)
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

