# This file is a part of the topology optimization program by Moise Rousseau

import numpy as np
import h5py
from ..PFLOTRAN import default_water_density, default_gravity, default_viscosity


class p_Weighted_Sum_Flux:
  """
Compute the sum of the squared flux through the connection of the cell given and
weighted by the material parameter.
In practice, minimise the squared mean flux in material designed by p=1.
  """
  def __init__(self, cell_ids_to_consider=None):
    """
    Default is to sum on all parametrized cell
    """
    #objective argument
    if cell_ids_to_consider is not None: 
      self.cell_ids_to_consider = np.array(cell_ids_to_consider) #PFLOTRAN id
    else: 
      self.cell_ids_to_consider = None
                            
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
    self.dobj_dmat_props = [0.,0.,0.,0.,0.,0.,0.] #derivative of the function wrt mat properties
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
    self.ids_p = -np.ones(np.max(p_ids),dtype='i8') #-1 mean not optimized
    self.ids_p[self.p_ids-1] = np.arange(len(self.p_ids))
    #here, we can assess the index in p of optimized PFLOTRAN cell with 
    # p_index = self.ids_p[PFLOTRAN_index-1]
    if self.cell_ids_to_consider is None: #sum on all parametrized cell
      self.cell_ids_to_consider = p_ids
    else: #check if all the cell to consider are parametrized (p is defined)
      mask = np.isin(self.cell_ids_to_consider, self.p_ids)
      if False in self.cell_ids_to_consider:
        print("Error! Some cell to sum the p-weighted flux are not parametrized (p is not defined at these cell):")
        print(self.cell_ids_to_consider[mask])
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
    return X_n1 * X_n2 / (self.d_fraction * X_n2 + (1-self.d_fraction) * X_n1)
  
  
  def get_connection_ids_to_sum(self):
    """
    Return a list of the connection ids to sum
    Note: internal connection are duplicated
    """
    if not self.initialized: self.__initialize__()
    ids_to_sum = np.append(self.connection_ids[self.connection_is_to_sum_1],
                           self.connection_ids[self.connection_is_to_sum_2], axis=0)
    return ids_to_sum
    
    
  def get_flux_at_connection(self, squared=True):
    """
    Compute the flux through each grid connection
    """
    k_con = self.interpole_at_face(self.k)
    d_P_con = self.pressure[self.connection_ids[:,0]-1] - \
                             self.pressure[self.connection_ids[:,1]-1]
    eg = default_water_density * default_gravity 
    if squared:
      flux_con = self.areas * k_con / default_viscosity * \
                             ( (d_P_con - eg * self.d_z_con) / self.distance)**2
    else:
      flux_con = self.areas * k_con / default_viscosity * \
                                (d_P_con - eg * self.d_z_con) / self.distance
    return flux_con

  
  ### COST FUNCTION ###
  def evaluate(self, p):
    """
    Evaluate the cost function
    """
    if not self.initialized: self.__initialize__()
    flux_con = self.get_flux_at_connection()
    flux_sum = np.sum(p[self.connections1_to_p] * flux_con[self.connection_is_to_sum_1])
    flux_sum += np.sum(p[self.connections2_to_p] * flux_con[self.connection_is_to_sum_2])
    return flux_sum
  
  
  ### PARTIAL DERIVATIVES ###
  def d_objective_dP(self,p): 
    """
    Evaluate the derivative of the function according to the pressure.
    Must have the size of the input problem
    """
    #initialize
    if not self.initialized: self.__initialize__()
    if self.dobj_dP is None:
      self.dobj_dP = np.zeros(len(self.pressure), dtype='f8') #pressure size of the grid
    else:
      self.dobj_dP[:] = 0.
    #compute derivative at all the connection
    d_flux_con = 2 * self.get_flux_at_connection(False) / self.distance
    for icon,ids in enumerate(self.connection_ids):
      if self.connection_is_to_sum_1[icon]:
        self.dobj_dP[ids[0]-1] += d_flux_con[icon] * p[self.ids_p[ids[0]-1]]
        self.dobj_dP[ids[1]-1] -= d_flux_con[icon] * p[self.ids_p[ids[0]-1]]
      if self.connection_is_to_sum_2[icon]: #bc
        self.dobj_dP[ids[0]-1] += d_flux_con[icon] * p[self.ids_p[ids[1]-1]]
        self.dobj_dP[ids[1]-1] -= d_flux_con[icon] * p[self.ids_p[ids[1]-1]]
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
    deriv = np.zeros(len(p),dtype='f8')
    K1 = self.k[self.connection_ids[:,0]-1] 
    K2 = self.k[self.connection_ids[:,1]-1]
    den = (self.d_fraction*K1 + (1-self.d_fraction)*K2)**2
    d_P_con = self.pressure[self.connection_ids[:,0]-1] - \
                             self.pressure[self.connection_ids[:,1]-1]
    prefactor = self.areas *  (d_P_con - default_water_density * \
                   default_gravity * self.d_z_con)**2 / (self.distance**2 * default_viscosity)
    dK1 = prefactor * K2**2 * (1-self.d_fraction) / den 
    dK2 = prefactor * K1**2 * self.d_fraction / den
    #build derivative
    for icon,ids in enumerate(self.connection_ids):
      i,j = -1, -1
      if ids[0] <= len(self.ids_p): i = self.ids_p[ids[0]-1]
      if ids[1] <= len(self.ids_p): j = self.ids_p[ids[1]-1]
      if self.connection_is_to_sum_1[icon]:
        if i != -1: deriv[i] += dK1[icon] * p[i]
        if j != -1: deriv[j] += dK2[icon] * p[j]
      if self.connection_is_to_sum_2[icon]: 
        if i != -1: deriv[i] += dK1[icon] * p[i]
        if j != -1: deriv[j] += dK2[icon] * p[j]
    self.dobj_dmat_props[2] = deriv
    return None
  
  def d_objective_dp_partial(self, p): 
    """
    PARTIAL Derivative of the function wrt the material parameter p (in input)
    """
    if not self.initialized: self.__initialize__()
    if self.dobj_dp_partial is None:
      self.dobj_dp_partial = np.zeros(len(p),dtype='f8')
    else:
      self.dobj_dp_partial[:] = 0.
    #compute flux at connection
    flux_con = self.get_flux_at_connection()
    for icon,ids in enumerate(self.connection_ids):
      if ids[0] <= len(self.ids_p): 
        i = self.ids_p[ids[0]-1]
        if i != -1: self.dobj_dp_partial[i] += flux_con[icon]
      if ids[1] <= len(self.ids_p): 
        j = self.ids_p[ids[1]-1]
        if j != -1: self.dobj_dp_partial[j] += flux_con[icon]
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
  
  
  ### WRAPPER FOR NLOPT ###
  def nlopt_optimize(self,p,grad):
    """
    Wrapper to evaluate and compute the derivative of the cost function
    for calling in nlopt
    """
    #could be used as is
    cf = self.evaluate(p)
    print(f"Current {self.name}: {cf:.6e}")
    if grad.size > 0:
      self.d_objective_dp_total(p,grad)
      print(f"Min gradient: {np.min(grad):.6e} at cell id {np.argmin(grad)}")
      print(f"Max gradient: {np.max(grad):.6e} at cell id {np.argmax(grad)}")
    return cf
  
  
  ### INITIALIZER FUNCTION ###
  def __initialize__(self):
    """
    Initialize the derived quantities from the inputs that must be compute one 
    time only.
    """
    self.initialized = True
    #found the connections belonging to the cell ids to consider
    mask = np.isin(self.connection_ids,self.cell_ids_to_consider)
    self.connection_is_to_sum_1 = mask[:,0] #PFLOTRAN indexing
    self.connection_is_to_sum_2 = mask[:,1]
    #connection cell ids to index in p
    self.connections1_to_p = self.ids_p[self.connection_ids[self.connection_is_to_sum_1][:,0]-1]
    self.connections2_to_p = self.ids_p[self.connection_ids[self.connection_is_to_sum_2][:,1]-1]
    #constant d_z_con
    self.d_z_con = self.z[self.connection_ids[:,0]-1] - self.z[self.connection_ids[:,1]-1] 
    return
  
  ### REQUIRED FOR CRAFTING ###
  def __require_adjoint__(self): return "RICHARDS"
  def __get_PFLOTRAN_output_variable_needed__(self):
    return self.output_variable_needed 
  def __get_name__(self): return self.name

                      
