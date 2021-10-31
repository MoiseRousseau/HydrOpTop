# This file is a part of the topology optimization program by Moise Rousseau

import numpy as np
import h5py
from ..Solver.PFLOTRAN import default_water_density, default_gravity, default_viscosity
from .common import __cumsum_from_connection_to_array__


class p_Weighted_Sum_Flux:
  """
Compute the sum of the squared flux through the connection of the cell given and
weighted by the material parameter.
In practice, minimise the squared mean flux in material designed by p=1.
Can inverse weighting using invert_weighting=True. Therefore, minimise the squared
flux in material designed by p=0
  """
  def __init__(self, cell_ids_to_consider=None, invert_weighting=False):
    """
    Default is to sum on all parametrized cell
    """
    #objective argument
    if cell_ids_to_consider is not None: 
      self.cell_ids_to_consider = np.array(cell_ids_to_consider) #PFLOTRAN id
    else: 
      self.cell_ids_to_consider = None
    self.invert_weighting = invert_weighting
                            
    #inputs for function evaluation
    self.d_fraction = None
    self.K = None
    self.pressure = None
    self.distance = None
    self.areas = None
    self.connection_ids = None
    
    #quantities derived from the input calculated one time
    self.initialized = False #a flag indicating calculated (True) or not (False)
    self.d_z_con = False
    self.connection_is_to_sum_1 = None  #PFLOTRAN indexing
    self.connection_is_to_sum_2 = None  #PFLOTRAN indexing
    self.connections1_to_p = None
    self.connections2_to_p = None
    
    #required attribute
    self.p_ids = None #correspondance between index in p and PFLOTRAN cell ids
                      #set by the crafter, user don't need this
    self.ids_p = None
    self.dobj_dP = None #derivative of the function wrt pressure
                      #to be passed to Adjoint class
    self.dobj_dmat_props = [0.,0.,None,0.,0.,0.,0.] #derivative of the function wrt mat properties
                                   #to be passed to Adjoint class
                                   #same size than self.output_variable_needed (see below)
    self.dobj_dp_partial = None #derivative of the function wrt material parameter
    self.adjoint = None #attribute storing adjoint
    
    #required for problem crafting
    self.output_variable_needed = ["LIQUID_PRESSURE", "FACE_AREA",
                                   "PERMEABILITY", "FACE_UPWIND_FRACTION", 
                                   "FACE_DISTANCE_BETWEEN_CENTER",
                                   "Z_COORDINATE", "CONNECTION_IDS"] 
    self.name = "p-Weighted Squared Flux Sum"
    return
    
  def set_inputs(self, inputs):
    """
    Method required by the problem crafter to pass the pflotran output
    variables to the objective
    Inputs argument have the same size and in the same order given in
    "self.output_variable_needed".
    """
    #parse input at each iteration
    no_bc_connections = (inputs[6][:,0] > 0) * (inputs[6][:,1] > 0)
    self.pressure = inputs[0]
    self.areas = inputs[1][no_bc_connections]
    self.k = inputs[2]
    self.d_fraction = inputs[3][no_bc_connections]
    self.distance = inputs[4][no_bc_connections]
    self.z = inputs[5]
    self.connection_ids = inputs[6][no_bc_connections]
    return
  
  def get_inputs(self):
    """
    Method that return the inputs in the same order than in "self.output_variable_needed".
    Required to pass the verification test
    """
    return [self.pressure, self.areas, self.k, self.d_fraction, self.distance,
            self.z, self.connection_ids]
  
  def set_p_to_cell_ids(self,p_ids):
    """
    Method that pass the correspondance between the index in p and the
    PFLOTRAN cell ids.
    Required by the Crafter
    """
    self.p_ids = p_ids #p to PFLOTRAN index
    if self.cell_ids_to_consider is None: #sum on all parametrized cell
      self.cell_ids_to_consider = p_ids
    else: #check if all the cell to consider are parametrized (p is defined)
      #mask = np.isin(self.cell_ids_to_consider, self.p_ids)
      if False in self.cell_ids_to_consider:
        mask = np.isin(self.cell_ids_to_consider, self.p_ids)
        print("Error! Some cell to sum the p-weighted flux are not parametrized (p is not defined at these cell):")
        print(self.cell_ids_to_consider[~mask])
        exit(1)
    return 
    
  def set_adjoint_problem(self, x):
    self.adjoint = x
    return
    
  def interpole_at_face(self, X):
    """
    Interpole the given cell centered variable at face
    Compute only the value at the connection in self.connections
    """
    X_n1 = X[self.connection_ids[:,0]-1]
    X_n2 = X[self.connection_ids[:,1]-1]
    return X_n1 * X_n2 / ((1-self.d_fraction) * X_n2 + self.d_fraction * X_n1)
  
  def get_connection_ids_to_sum(self):
    """
    Return a list of the connection ids to sum
    Note: internal connection are duplicated
    """
    if not self.initialized: self.__initialize__()
    ids_to_sum = np.append(self.connection_ids[self.connection_is_to_sum_1],
                           self.connection_ids[self.connection_is_to_sum_2], axis=0)
    return ids_to_sum


  
  ### COST FUNCTION ###
  def evaluate(self, p):
    """
    Evaluate the cost function
    """
    if not self.initialized: self.__initialize__()
    if self.invert_weighting: pp = 1 - p
    else: pp = p
    k_con = self.interpole_at_face(self.k)
    d_P_con = self.pressure[self.connection_ids[:,1]-1] - \
                             self.pressure[self.connection_ids[:,0]-1]
    eg = default_water_density * default_gravity 
    flux_con = self.areas * k_con / default_viscosity * \
                           ( (d_P_con + eg * self.d_z_con) / self.distance ) ** 2
    flux_sum = np.sum( (pp[self.connections1_to_p] * flux_con)[self.connection_is_to_sum_1] )
    flux_sum += np.sum( (pp[self.connections2_to_p] * flux_con)[self.connection_is_to_sum_2] )
    return flux_sum
  
  
  ### PARTIAL DERIVATIVES ###
  def d_objective_dP(self,p): 
    """
    Evaluate the derivative of the function according to the pressure.
    Must have the size of the input problem
    """
    #initialize
    if not self.initialized: self.__initialize__()
    if self.invert_weighting: pp = 1 - p
    else: pp = p
    if self.dobj_dP is None:
      self.dobj_dP = np.zeros(len(self.pressure), dtype='f8') #pressure size of the grid
    else:
      self.dobj_dP[:] = 0.
    #compute derivative at all the connection
    k_con = self.interpole_at_face(self.k)
    d_P_con = self.pressure[self.connection_ids[:,1]-1] - \
                             self.pressure[self.connection_ids[:,0]-1]
    eg = default_water_density * default_gravity
    d_flux_con =  -2. * self.areas * k_con / default_viscosity * \
                                   (d_P_con + eg * self.d_z_con) / self.distance**2
    p_ = np.where(self.connection_is_to_sum_1, pp[self.connections1_to_p], 0.)
    p_ += np.where(self.connection_is_to_sum_2, pp[self.connections2_to_p], 0.)
    d_flux_con *= p_
    
    __cumsum_from_connection_to_array__(self.dobj_dP, 
                                        self.connection_ids[:,0]-1, 
                                        d_flux_con)
    __cumsum_from_connection_to_array__(self.dobj_dP, self.connection_ids[:,1]-1,
                                        -d_flux_con)
    return
    
  
  def d_objective_d_mat_props(self, p):
    """
    Derivative of the function according to input variable
    Must have the size of the input problem and ordered as in PFLOTRAN output, i.e.
    dc/dXi[0] => cell_id = 1
    dc/dXi[1] => cell_id = 2
    ...
    Argument:
    - p : the material parameter
    Return:
    - A list of the derivative in the same order they are listed in 
      self.mat_props_dependance
    No need to output the derivative according to pressure (it's d_objective_dP task)
    Note: if the variable is not parametrized in the Materials module, the derivative is 
    not considered and therefore could be zero
    """
    #initialize
    if not self.initialized: self.__initialize__()
    if self.invert_weighting: pp = 1 - p
    else: pp = p
    K1 = self.k[self.connection_ids[:,0]-1] 
    K2 = self.k[self.connection_ids[:,1]-1]
    den = (self.d_fraction*K1 + (1-self.d_fraction)*K2)**2
    d_P_con = self.pressure[self.connection_ids[:,1]-1] - \
                             self.pressure[self.connection_ids[:,0]-1]
    eg = default_water_density * default_gravity
    prefactor = self.areas / default_viscosity * \
                             ( (d_P_con + eg * self.d_z_con) / self.distance ) ** 2
    
    dK1 = prefactor * K2**2 * (1-self.d_fraction) / den
    dK2 = prefactor * K1**2 * self.d_fraction / den
    
    #build derivative
    if self.dobj_dmat_props[2] is None:
      self.dobj_dmat_props[2] = np.zeros(len(self.pressure),dtype='f8')
    else: 
      self.dobj_dmat_props[2][:] = 0.
      
    p_ = np.where(self.connection_is_to_sum_1, pp[self.connections1_to_p], 0.)
    p_ += np.where(self.connection_is_to_sum_2, pp[self.connections2_to_p], 0.)
    
    __cumsum_from_connection_to_array__(self.dobj_dmat_props[2], 
                                    self.connection_ids[:,0]-1, p_*dK1)
    __cumsum_from_connection_to_array__(self.dobj_dmat_props[2], 
                                    self.connection_ids[:,1]-1, p_*dK2)
    return None
  
  
  def d_objective_dp_partial(self, p): 
    """
    PARTIAL Derivative of the function wrt the material parameter p (in input)
    """
    if not self.initialized: self.__initialize__()
    if self.invert_weighting: factor = -1.
    else: factor = 1.
    if self.dobj_dp_partial is None:
      self.dobj_dp_partial = np.zeros(len(p),dtype='f8')
    else:
      self.dobj_dp_partial[:] = 0.
    #compute flux at connection
    k_con = self.interpole_at_face(self.k)
    d_P_con = self.pressure[self.connection_ids[:,1]-1] - \
                             self.pressure[self.connection_ids[:,0]-1]
    eg = default_water_density * default_gravity 
    flux_con = factor * self.areas * k_con / default_viscosity * \
                          ( (d_P_con + eg * self.d_z_con) / self.distance ) ** 2
    __cumsum_from_connection_to_array__(self.dobj_dp_partial, 
                                        self.connections1_to_p, flux_con)
    __cumsum_from_connection_to_array__(self.dobj_dp_partial, 
                                        self.connections2_to_p, flux_con)
    
    return 
  
  
  ### TOTAL DERIVATIVE ###
  def d_objective_dp_total(self, p, out=None): 
    """
    Evaluate the TOTAL derivative of the function according to the density
    parameter p. If a numpy array is provided, derivative will be copied 
    in this array, else create a new numpy array.
    """
    if not self.initialized: self.__initialize__()
    #this method could be used as is
    if out is None:
      out = np.zeros(len(self.p), dtype='f8')
    self.d_objective_dP(p) #update function derivative wrt pressure
    self.d_objective_d_mat_props(p) #update function derivative wrt mat prop
    self.d_objective_dp_partial(p) #update function derivative wrt mat parameter p
    out[:] = self.adjoint.compute_sensitivity(p, self.dobj_dP, 
                 self.dobj_dmat_props, self.output_variable_needed) + self.dobj_dp_partial
    return out
  
  
  
  ### INITIALIZER FUNCTION ###
  def __create_indexes_for_reduce_at__(self, connection_id, con_type):
    """
    Sort the connection to sum and create the array of indexes for reduceat()
    """
    sorted_index = np.argsort( connection_id[con_type] )
    non_param = np.sum( connection_id[con_type] == -1 )
    if non_param:
      temp_array = np.bincount( connection_id[con_type][sorted_index][non_param:] )
    else:
      temp_array = np.bincount( connection_id[con_type][non_param:] )
    if len(temp_array) > 0:
      temp_array = temp_array[temp_array > 0]
      index_reduce_at = np.zeros(len(temp_array),dtype='i8')
      index_reduce_at[1:] = np.cumsum(temp_array)[:-1]
      index_reduce_at += non_param
    else:
      index_reduce_at = None
    return sorted_index, index_reduce_at
  
  
  def __initialize__(self):
    """
    Initialize the derived quantities from the inputs that must be compute one 
    time only.
    """
    self.initialized = True
    #build correspondance between the cell ids and index in p
    #here, we can assess the index in p of optimized PFLOTRAN cell with 
    # p_cell_i = self.ids_p[cell_i-1]
    self.ids_p = -np.ones(np.max(self.connection_ids),dtype='i8') #-1 mean not optimized
    self.ids_p[self.p_ids-1] = np.arange(len(self.p_ids))
    #found the connections belonging to the cell ids to consider
    mask = np.isin(self.connection_ids,self.cell_ids_to_consider)
    self.connection_is_to_sum_1 = mask[:,0] #PFLOTRAN indexing
    self.connection_is_to_sum_2 = mask[:,1]
    #connection cell ids to index in p
    self.connections1_to_p = self.ids_p[self.connection_ids[:,0]-1]
    self.connections2_to_p = self.ids_p[self.connection_ids[:,1]-1]
    #constant d_z_con
    self.d_z_con = self.z[self.connection_ids[:,1]-1] - self.z[self.connection_ids[:,0]-1]
    return
  
  ### REQUIRED FOR CRAFTING ###
  def __require_adjoint__(self): return "RICHARDS"
  def __get_PFLOTRAN_output_variable_needed__(self):
    return self.output_variable_needed 
  def __get_name__(self): return self.name

                      
