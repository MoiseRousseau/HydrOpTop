import h5py
import numpy as np
import nlopt

from HydrOpTop.Adjoints import Sensitivity_Richards


class Steady_State_Crafter:
  """
  Craft a topology optimization problem in steady state
  Argument:
  - mat_props: a list of material properties that vary with the density
               parameter p (Material classes instances)
  - solver: object that manage the PDE solver (PFLOTRAN class instance)
  - objectif: the objective function (Objectif class instance)
  - constrains: a list of constrains (Constrain class instances
  - coupling: specify how each material properties should be optimized
                       (coupled = one unique p parameter per cell, half = coupled
                       for duplicate ids to optimize in each material, none =
                       each material properties have a separate parameter) 
                       (default=total)
  """
  def __init__(self, objectif, solver, mat_props, constrains, coupling="total"):
    self.mat_props = mat_props
    self.solver = solver
    self.obj = objectif
    self.constrains = constrains
    self.coupling = coupling
    
    #self.Xi = None #store material properties
    self.adjoint = None
    self.Yi = None #store PFLOTRAN output
    self.p_ids = None
    
    #option
    self.print_every = 0
    self.print_every_out = "p.h5"
    
    self.__initialize_IO_array__()
    self.first_call = True
    self.func_eval = 0
    self.last_p = None
    self.adjoint_algo = None
    return
  
  def get_problem_size(self): return self.problem_size
  
  def set_adjoint_problem_algo(self, algo):
    self.adjoint_algo = algo
    return
  
  def print_density_parameter_every_iteration(self, every_it, out=None):
    self.print_every = every_it
    if out is not None: self.print_every_out = out
    #TODO implement this
    return
  
  
  
  
  def evaluate_objective(self, p):
    ### UPDATE MAT PROPERTIES AND RUN PFLOTRAN ###
    #Given p, update material properties
    for mat_prop in self.mat_props:
      X = mat_prop.convert_p_to_mat_properties(p)
      self.solver.create_cell_indexed_dataset(X, mat_prop.get_name().lower(),
                    X_ids=mat_prop.get_cell_ids_to_parametrize(), resize_to=True)
    #run PFLOTRAN
    ret_code = self.solver.run_PFLOTRAN()
    if ret_code: return np.nan
    
    ### GET PFLOTRAN OUTPUT ###
    for i,var in enumerate(self.obj.__get_PFLOTRAN_output_variable_needed__()):
      self.solver.get_output_variable(var, self.Yi[i], -1) #last timestep
      
    ### UPDATE CONSTRAINS ###
    count = 0
    for constrain in self.constrains:
      for var in constrain.__get_PFLOTRAN_output_variable_needed__():
        self.solver.get_output_variable(var, self.constrain_arrays[count], -1)
        count += 1
    
    ### EVALUATE COST FUNCTION AND ITS DERIVATIVE ###
    # note that we have in place assignement, so we don't have to
    # update the Yi in the objective
    cf = self.obj.evaluate()
    return cf
  
  
  
  def compute_gradient(self, p, grad=None):
    #note: the evaluate_objective function should have been must 
    # have been called before
    if grad is None:
      grad = np.zeros(len(self.p_ids),dtype='f8')
    ### CREATE ADJOINT ###
    if self.first_call:
      self.fist_call = False
      self.__initialize_adjoint__()
    ### UPDATE ADJOINT ###
    #cost derivative to pressure (vector)
    self.obj.d_objective_dP(self.adjoint.dc_dP)
    #cost derivative to mat properties (vector)
    for i,mat_prop in enumerate(self.mat_props):
      var = mat_prop.get_name()
      if self.obj.__depend_of_mat_props__(var):
        self.obj.d_objective_d_inputs(var, self.adjoint.dc_dXi[i].data)
    #update matrix
    #note the I,J do not change, only the data
    #residual according to pressure
    self.solver.update_sensitivity("LIQUID_PRESSURE", self.adjoint.dR_dP)
    #residual according to mat_prop
    for i,mat_prop in enumerate(self.mat_props):
      self.solver.update_sensitivity(mat_prop.get_name(),
                                    self.adjoint.dR_dXi[i])
    #material property deriv according to mat parameter
    for i,mat_prop in enumerate(self.mat_props):
      mat_prop.d_mat_properties(p, self.adjoint.dXi_dp[i].data)

    ### COMPUTE ADJOINT ###
    self.adjoint.compute_sensitivity(grad, assign_at_ids=self.p_ids)
    return grad
    
    
  ### READY MADE WRAPPER FOR POPULAR LIBRARY ###
  def nlopt_function_to_optimize(self, p, grad):
    # IO stuff
    self.func_eval += 1
    print(f"\nFonction evaluation {self.func_eval}")
    # objective
    cf = self.evaluate_objective(p)
    print(f"Current cost function: {cf}")
    if grad.size > 0:
      self.compute_gradient(p,grad)
    self.last_p = np.copy(p)
    return cf
    
  def scipy_function_to_optimize():
    return
  
  
  
  def __initialize_IO_array__(self):
    #create material parameter p
    if self.coupling == "total":
      #verify if each ids are the same
      X = self.mat_props[0].get_cell_ids_to_parametrize()
      if len(self.mat_props) > 1:
        for x in self.mat_props:
          if x.get_ids_to_optimize() != X: 
            print("Different cell ids to optimize")
            print("Can not use 'total' coupling method")
            exit(1)
      if X is None: 
        self.problem_size = self.solver.n_cells
        self.p_ids = np.arange(1, self.problem_size+1)
      else: 
        self.problem_size = len(X)
        self.p_ids = X
    #elif self.coupling == "none":
    #  self.Xi = 
    else:
      print("Error: Other coupling method not yet implemented")
      exit(1)
    
    #initialize output
    n_outputs = len(self.obj.__get_PFLOTRAN_output_variable_needed__())
    self.Yi = [np.zeros(self.solver.n_cells, dtype='f8') for x in range(n_outputs)]
    self.obj.set_inputs(self.Yi)
    
    #initialize constrains
    self.constrain_arrays = []
    for constrain in self.constrains:
      for i,dep in enumerate(constrain.__get_PFLOTRAN_output_variable_needed__()):
        self.constrain_arrays.append(np.zeros(self.solver.n_cells, dtype='f8'))
      constrain.set_inputs(self.constrain_arrays[-i-1:])
      if constrain.__need_p_cell_ids__():
        constrain.set_p_cell_ids(self.p_ids)
    return
  
  
  
  def __initialize_adjoint__(self):
    #cost derivative to pressure (vector)
    dc_dP = np.zeros(self.solver.n_cells, dtype='f8')
    #cost derivative to mat properties (vector)
    if self.obj.__depend_of_mat_props__(): 
      dc_dXi = np.zeros(self.solver.n_cells, dtype='f8')
    else:
      dc_dXi = None
    #mat properties derivative to parameter (diag matrix)
    dXi_dp = [np.zeros(self.solver.n_cells, dtype='f8') 
                                 for mat_prop in self.mat_props]
    #residual derivtive to mat prop (matrix)
    dR_dXi = []
    for mat_prop in self.mat_props:
      dR_dXi.append(self.solver.get_sensitivity(mat_prop.get_name()))
    #residual derivative to pressure
    dR_dP = self.solver.get_sensitivity("LIQUID_PRESSURE")
    self.adjoint = Sensitivity_Richards(dc_dP, dXi_dp, dc_dXi, dR_dXi, dR_dP)
    if self.adjoint_algo is not None:
      self.adjoint.set_adjoint_solving_algo(self.adjoint_algo)
    return
    

