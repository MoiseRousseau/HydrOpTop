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
  - filter: the filter to be used to relate the density to a filtered density
            (a fitler class instance) (None by default).
  """
  def __init__(self, objectif, solver, mat_props, constrains, filter=None, coupling="total"):
    self.mat_props = mat_props
    self.solver = solver
    self.obj = objectif
    self.constrains = constrains
    self.filter = filter
    self.coupling = coupling
    
    #self.Xi = None #store material properties (solver inputs)
    self.Yi = None #store solver outputs
    self.filter_i = None #filter inputs
    self.p_ids = None #correspondance between p index and cell ids in the solver
                      #i.e. p[0] parametrize cell X, p[1] cell Y, ...
    
    self.adjoint_algo = None
    self.adjoint_tol = None
    
    #option
    self.print_every = 0
    self.print_every_out = "p.h5"
    
    self.__initialize_IO_array__()
    self.__initialize_filter__()
    self.first_call_evaluation = True
    self.first_call_gradient = True
    self.func_eval = 0
    self.last_p = None
    return
  
  def get_problem_size(self): return self.problem_size
  
  def set_adjoint_problem_algo(self, algo, tol=None):
    if self.obj.__require_adjoint__():
      self.obj.adjoint.set_adjoint_solving_algo(algo, tol)
    for constrain in self.constrains:
      if constrain.__require_adjoint__():
        constrain.adjoint.set_adjoint_solving_algo(algo, tol)
    return
  
  def print_density_parameter_every_iteration(self, every_it, out=None):
    self.print_every = every_it
    if out is not None: self.print_every_out = out
    #TODO implement this
    return
  
  
  
  def pre_evaluation_objective(self, p):
    ###FILTERING: convert p to p_bar
    if self.filter is None:
      p_bar = p
    else: 
      for i,var in enumerate(self.filter.__get_PFLOTRAN_output_variable_needed__()):
        self.solver.get_output_variable(var, self.filter_i[i], -1) #last timestep
      p_bar = self.filter.get_filtered_density(p)
    
    ### UPDATE MAT PROPERTIES AND RUN PFLOTRAN ###
    #Given p, update material properties
    for mat_prop in self.mat_props:
      X = mat_prop.convert_p_to_mat_properties(p_bar)
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
        self.solver.get_output_variable(var, self.constrain_inputs_arrays[count], -1)
        count += 1
    return
    
    
  ### READY MADE WRAPPER FOR POPULAR LIBRARY ###
  def nlopt_function_to_optimize(self, p, grad):
    # IO stuff
    self.func_eval += 1
    print(f"\nFonction evaluation {self.func_eval}")
    ### PRE-EVALUATION ###
    self.pre_evaluation_objective(p)
    ### OBJECTIVE EVALUATION AND DERIVATIVE ###
    cf = self.obj.nlopt_optimize(p,grad)
    self.last_p = np.copy(p)
    return cf
    
  def scipy_function_to_optimize():
    return
  
  
  
  def __initialize_IO_array__(self):
    print("Initialization...")
    #verify if each cells to parametrize are the same
    X = self.mat_props[0].get_cell_ids_to_parametrize()
    if len(self.mat_props) > 1:
      for x in self.mat_props:
        if x.get_ids_to_optimize() != X: 
          print("Different cell ids to optimize")
          print("HydrOpTop require the different mat properties to parametrize the same cell ids")
          exit(1)
    #create correspondance and problem size
    if X is None: #i.e. parametrize all cell in the simulation
      self.problem_size = self.solver.n_cells
      self.p_ids = None#np.arange(0, self.problem_size) 
    else: 
      self.problem_size = len(X)
      self.p_ids = X #from 0 based indexing to PFLOTRAN indexing
    
    #initialize solver output for objective function
    n_outputs = len(self.obj.__get_PFLOTRAN_output_variable_needed__())
    self.Yi = [np.zeros(self.solver.n_cells, dtype='f8') for x in range(n_outputs)]
    self.obj.set_inputs(self.Yi)
    #initialize adjoint for objective function
    if self.obj.__require_adjoint__():
      which = self.obj.__require_adjoint__()
      if which == "RICHARDS":
        adjoint = Sensitivity_Richards(self.mat_props, self.solver, self.p_ids)
        self.obj.set_adjoint_problem(adjoint)
    self.obj.set_p_cell_ids(self.p_ids)
    if filter:
      self.obj.set_filter(self.filter)
    
    #initialize constrains
    #TODO: initialize constrains
    #TODO: change self.p_ids that go through 0 and not 1
    self.constrain_inputs_arrays = []
    for constrain in self.constrains:
      for i,dep in enumerate(constrain.__get_PFLOTRAN_output_variable_needed__()):
        self.constrain_inputs_arrays.append(np.zeros(self.solver.n_cells, dtype='f8'))
      constrain.set_inputs(self.constrain_inputs_arrays[-i-1:])
      if constrain.__require_adjoint__():
        which = self.obj.__require_adjoint__()
        if which == "RICHARDS":
          adjoint = Sensitivity_Richards(self.mat_props, self.solver, self.p_ids)
          constrain.set_adjoint_problem(adjoint)
      constrain.set_p_cell_ids(self.p_ids)
      if self.filter:
        constrain.set_filter(self.filter)
    return
    
  
  def __initialize_filter__(self):
    print("Filter initialization")
    #filter initialization is tricky, because it may need PFLOTRAN output variable
    # that have not been created yet. Thus, for instance, we have to run
    # PFLOTRAN one time to initialize the filter (even if it's costly)...
    if self.filter is None:
      return
      
    self.filter.set_p_cell_ids(self.p_ids)
    n_inputs = len(self.filter.__get_PFLOTRAN_output_variable_needed__())
    
    #TODO: refactor below as I don't like to make one iteration just for
    # initalize the filter
    if n_inputs > 0:
      #run pflotran to get its output
      
      #Given p, update material properties
      p_bar = np.zeros(len(self.p_ids), dtype='f4')
      for mat_prop in self.mat_props:
        X = mat_prop.convert_p_to_mat_properties(p_bar)
        self.solver.create_cell_indexed_dataset(X, mat_prop.get_name().lower(),
                      X_ids=mat_prop.get_cell_ids_to_parametrize(), resize_to=True)
      #run PFLOTRAN
      ret_code = self.solver.run_PFLOTRAN() #TODO: uncomment
    
    self.filter_i = [np.zeros(self.solver.n_cells, dtype='f8') for x in range(n_inputs)]
    for i,var in enumerate(self.filter.__get_PFLOTRAN_output_variable_needed__()):
      self.solver.get_output_variable(var, self.filter_i[i], -1) #last timestep
    self.filter.set_inputs(self.filter_i)
    self.filter.initialize()
    self.d_pbar_dp = self.filter.get_filter_derivative(p_bar)
    return
    

