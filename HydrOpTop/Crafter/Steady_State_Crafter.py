import h5py
import numpy as np
import nlopt
import functools

from HydrOpTop.Adjoints import Sensitivity_Richards
from HydrOpTop.IO import IO


class Steady_State_Crafter:
  """
  Craft a topology optimization problem in steady state
  Argument:
  - mat_props: a list of material properties that vary with the density
               parameter p (Material classes instances)
  - solver: object that manage the PDE solver (PFLOTRAN class instance)
  - objective: the objective function (Function class instance)
  - constraints: a list of constraints (Function class instances
  - filter: the filter to be used to relate the density to a filtered density
            (a fitler class instance) (None by default).
  """
  def __init__(self, objective, solver, mat_props, constraints, filter=None, io=None):
    self.__print_information__()
    self.mat_props = mat_props
    self.solver = solver
    self.obj = objective
    self.constraints = constraints
    self.filter = filter
    if io is None:
      self.IO = IO()
    else:
      self.IO = io
    
    #self.Xi = None #store material properties (solver inputs)
    self.Yi = None #store solver outputs
    self.filter_i = None #filter inputs
    self.p_ids = None #correspondance between p index and cell ids in the solver
                      #i.e. p[0] parametrize cell X, p[1] cell Y, ...
    self.ids_p = None #correspondance between the cell ids in the solver and p
                      #i.e. cell ids x is parametrize by p index ids_p[x-1]
    self.constraint_inputs_arrays = None
    self.adjoint_algo = None
    self.adjoint_tol = None
    
    self.first_call_evaluation = True
    self.first_call_gradient = True
    self.func_eval = 0
    self.last_p = None
    self.first_cf = None
    self.first_p = None
    
    self.__initialize_IO_array__()
    self.__initialize_filter__()
    return
  
  def get_problem_size(self): 
    return self.problem_size
    
  
  def create_density_parameter_vector(self, init_val=0.):
    return np.zeros(self.problem_size, dtype='f8') + init_val
  
  def create_random_density_parameter_vector(self, bounds):
    m, M = bounds
    return (M - m) * np.random.random(self.problem_size) + m
    
  def simulation_ids_to_p(self, ids):
    """
    Return the correspondance between the given ids and the index in the
    density parameter created with the routine create_density_parameter_vector()
    """
    if self.ids_p is None:
      self.ids_p = -np.ones(self.solver.get_grid_size(), dtype='i8')
      self.ids_p[self.p_ids-1] = np.arange(len(self.p_ids))
    return self.ids_p[ids-1]
  
  
  ### PRE-EVALUATION ###
  def pre_evaluation_objective(self, p):
    ### UPDATE MAT PROPERTIES AND RUN PFLOTRAN ###
    #Given p, update material properties
    for mat_prop in self.mat_props:
      X = mat_prop.convert_p_to_mat_properties(p)
      self.solver.create_cell_indexed_dataset(X, mat_prop.get_name().lower(),
                    X_ids=mat_prop.get_cell_ids_to_parametrize(), resize_to=True)
    #run PFLOTRAN
    ret_code = self.solver.run()
    if ret_code: exit(ret_code)
    ### UPDATE OBJECTIVE ###
    if self.Yi is None: #need to initialize
      self.Yi = []
      for var in self.obj.__get_solved_variables_needed__():
        self.Yi.append(self.solver.get_output_variable(var))
      for var in self.obj.__get_input_variables_needed__():
        self.Yi.append(self.solver.get_output_variable(var))
      self.obj.set_inputs(self.Yi)
    else: 
      for i,var in enumerate(self.obj.__get_solved_variables_needed__()):
        self.solver.get_output_variable(var, self.Yi[i], -1) #last timestep
      for i,var in enumerate(self.obj.__get_input_variables_needed__()):
        self.solver.get_output_variable(var, self.Yi[i], -1) #last timestep
      
    ### UPDATE constraints ###
    if self.constraint_inputs_arrays is None: #need to initialize
      self.constraint_inputs_arrays = []
      for constraint in self.constraints:
        temp = []
        for var in constraint.__get_PFLOTRAN_output_variable_needed__():
          if var == "CONNECTION_IDS":
            temp.append(self.solver.get_internal_connections())
            continue
          temp.append(self.solver.get_output_variable(var))
        self.constraint_inputs_arrays.append(temp)
        constraint.set_inputs(self.constraint_inputs_arrays[-1])
    else: 
      for i,constraint in enumerate(self.constraints):
        for j,var in enumerate(constraint.__get_PFLOTRAN_output_variable_needed__()):
          if var == "CONNECTION_IDS":
            self.solver.get_internal_connections(self.constraint_inputs_arrays[i][j])
            continue
          self.solver.get_output_variable(var, self.constraint_inputs_arrays[i][j], -1)
    return
  
  
  ### OPTIMIZER ###
  def optimize(self, optimizer="nlopt-mma", 
                     initial_guess=None,
                     action="minimize", 
                     density_parameter_bounds=[0.001, 1],
                     tolerance_constraints=0.005,
                     max_it=50,
                     ftol=None):
    if optimizer == "nlopt-mma":
      algorithm = nlopt.LD_MMA #use MMA algorithm
      opt = nlopt.opt(algorithm, self.problem_size)
      if action == "minimize":
        opt.set_min_objective(self.nlopt_function_to_optimize) 
      elif action == "maximize":
        opt.set_max_objective(self.nlopt_function_to_optimize) 
      else:
        print(f"Error: Unknown action \"{action}\" (should be \"minimize\" or \"maximize\"")
        return None
        
      #add constraints
      for i in range(len(self.constraints)):
        opt.add_inequality_constraint(self.nlopt_constraint(i),
                                      tolerance_constraints)
      #define minimum and maximum bounds
      opt.set_lower_bounds(np.zeros(self.get_problem_size(), dtype='f8') +
                           density_parameter_bounds[0])
      opt.set_upper_bounds(np.zeros(self.get_problem_size(), dtype='f8') +
                           density_parameter_bounds[1])
      #define stop criterion
      opt.set_maxeval(max_it)
      if ftol is not None: 
        opt.set_ftol_rel(ftol)
      #initial guess
      if initial_guess is None:
        initial_guess = np.zeros(self.get_problem_size(), dtype='f8') + \
                           density_parameter_bounds[0]
      try:
        p_opt = opt.optimize(initial_guess)
        print(initial_guess)
      except(KeyboardInterrupt):
        p_opt = self.last_p
    else:
      print(f"Error: Unknown optimizer \"{optimizer}\"")
      return None
      
    self.IO.output(self.func_eval, 
                   self.last_cf, 
                   self.last_constraints, 
                   self.last_p, 
                   self.last_grad, 
                   [0], 
                   val_at=self.p_ids-1)
    print("END!")
    return p_opt
    
  
  
  ### READY MADE WRAPPER FOR POPULAR LIBRARY ###
  def nlopt_function_to_optimize(self, p, grad):
    """
    Cost function to pass to NLopt method "set_min/max_objective()"
    """
    #save last iteration
    if self.func_eval != 0:
      if self.filter is None: 
        self.IO.output(self.func_eval, 
                       self.last_cf, 
                       self.last_constraints, 
                       self.last_p, 
                       self.last_grad, 
                       [0], 
                       val_at=self.p_ids-1)
        #self.IO.output(self.func_eval, cf, constraints_val, p, grad_cf, grad_constraints, p_bar)
      else:
        self.IO.output(self.func_eval, 
                       self.last_cf, 
                       self.last_constraints, 
                       self.last_p, 
                       self.last_grad, 
                       [0], 
                       self.last_p_bar, 
                       val_at=self.p_ids-1)
    #start new it
    self.func_eval += 1
    print(f"\nFonction evaluation {self.func_eval}")
    ###FILTERING: convert p to p_bar
    if self.filter is None:
      p_bar = p
    else:
      for i,var in enumerate(self.filter.__get_PFLOTRAN_output_variable_needed__()):
        self.solver.get_output_variable(var, self.filter_i[i], -1) #last timestep
      p_bar = self.filter.get_filtered_density(p)
    if self.first_p is None: self.first_p = p_bar.copy()
    ### PRE-EVALUATION
    self.pre_evaluation_objective(p_bar)
    ### OBJECTIVE EVALUATION AND DERIVATIVE
    cf = self.obj.evaluate(p_bar)
    if grad.size > 0:
      dobj_dY = self.obj.d_objective_dY(p_bar)
      dobj_dp_partial = self.obj.d_objective_dp_partial(p_bar)
      dobj_dX = self.obj.d_objective_dX(p_bar)
      grad[:] = self.obj.adjoint.compute_sensitivity(p_bar, dobj_dY, 
                  dobj_dX, self.obj.__get_input_variables_needed__()) + \
                  dobj_dp_partial
    if self.first_cf is None: 
      self.first_cf = cf
    if self.filter and grad.size > 0:
      grad[:] = self.filter.get_filter_derivative(p).transpose().dot(grad)
    #save for output at next iteration
    self.last_cf = cf
    self.last_grad = grad
    self.last_p[:] = p
    self.last_p_bar = p_bar
    #normalize cf to 1
    cf /= self.first_cf
    grad /= self.first_cf
    #print to user
    print(f"Current {self.obj.name}: {cf*self.first_cf:.6e}")
    print(f"Min gradient: {np.min(grad*self.first_cf):.6e} at cell id {np.argmin(grad)+1}")
    print(f"Max gradient: {np.max(grad*self.first_cf):.6e} at cell id {np.argmax(grad)+1}")
    return cf
    
  
  def nlopt_constraint(self, i):
    """
    Function defining the ith constraints to pass to nlopt "set_(in)equality_constraint()"
    """
    return functools.partial(self.__nlopt_generic_constraint_to_optimize__, iconstraint=i)
  
  def __nlopt_generic_constraint_to_optimize__(self, p, grad, iconstraint=0):
    ###FILTERING: convert p to p_bar
    if self.filter is None:
      p_bar = p
    else: 
      for i,var in enumerate(self.filter.__get_PFLOTRAN_output_variable_needed__()):
        self.solver.get_output_variable(var, self.filter_i[i], -1) #last timestep
      p_bar = self.filter.get_filtered_density(p)
    constraint = self.constraints[iconstraint].evaluate(p_bar)
    if grad.size > 0:
      self.constraints[iconstraint].d_objective_dp_total(p_bar, grad)
    
      #dobj_dP = self.obj.d_objective_dP(p_bar)
      #dobj_dp_partial = self.obj.d_objective_dp_partial(p_bar)
      #dobj_dmat_props = self.obj.d_objective_d_mat_props(p_bar)
      #grad[:] = self.adjoint.compute_sensitivity(p_bar, dobj_dP, 
      #            dobj_dmat_props, self.obj.output_variable_needed) + \
      #            dobj_dp_partial
      
    if self.filter:
      grad[:] = self.filter.get_filter_derivative(p_bar).transpose().dot(grad)
    tol = self.constraints[iconstraint].__get_constraint_tol__()
    print(f"Current {self.constraints[iconstraint].name} constraint: {constraint+tol:.6e}")
    self.last_constraints[iconstraint] = constraint+tol
    return constraint
    
   
  def scipy_function_to_optimize():
    return
  
  
  
  
  ### INITIALIZATION ###
  def __initialize_IO_array__(self):
    print("Initialization...")
    #verify if each cells to parametrize are the same
    X = self.mat_props[0].get_cell_ids_to_parametrize()
    if len(self.mat_props) > 1:
      for x in self.mat_props:
        if x.get_ids_to_optimize() != X: 
          print("Different cell ids to optimize")
          print("HydrOpTop require the different mat properties to parametrize \
                 the same cell ids")
          exit(1)
    #create correspondance and problem size
    if X is None: #i.e. parametrize all cell in the simulation
      self.problem_size = self.solver.n_cells
      self.p_ids = np.arange(1, self.problem_size+1) 
    else: 
      self.problem_size = len(X)
      self.p_ids = X #from 0 based indexing to PFLOTRAN indexing
    self.last_p = np.zeros(len(self.p_ids),dtype='f8')
    
    #initialize solver output for objective function
    #do not set inputs array because we don't know the size of the connection_ids
    #in case of face output
    
    #initialize adjoint for objective function
    if self.obj.__require_adjoint__() and (self.obj.adjoint is None):
      which = self.obj.__require_adjoint__()
      if which == "RICHARDS":
        adjoint = Sensitivity_Richards(self.mat_props, self.solver, self.p_ids)
        self.obj.set_adjoint_problem(adjoint)
    self.obj.set_p_to_cell_ids(self.p_ids)
    
    #initialize adjoint for constraints
    self.last_constraints = [0. for x in self.constraints]
    for constraint in self.constraints:
      if constraint.__require_adjoint__() and (constraint.adjoint is None):
        which = self.obj.__require_adjoint__()
        if which == "RICHARDS":
          adjoint = Sensitivity_Richards(self.mat_props, self.solver, self.p_ids)
          constraint.set_adjoint_problem(adjoint)
      constraint.set_p_to_cell_ids(self.p_ids)
    #initialize IO
    self.IO.communicate_functions_names(self.obj.__get_name__(), 
         [x.__get_name__() for x in self.constraints])
    vertices, cells, indexes = self.solver.get_mesh()
    self.IO.set_mesh_info(vertices, cells, indexes,
                          self.solver.get_var_location())
    return
    
  
  def __initialize_filter__(self):
    print("Filter initialization")
    #filter initialization is tricky, because it may need PFLOTRAN output variable
    # that have not been created yet. Thus, for instance, we have to run
    # PFLOTRAN one time to initialize the filter (even if it's costly)...
    if self.filter is None:
      return
      
    self.filter.set_p_to_cell_ids(self.p_ids)
    n_inputs = len(self.filter.__get_PFLOTRAN_output_variable_needed__())
    
    #TODO: refactor below as I don't like to make one iteration just for
    # initalize the filter
    if n_inputs > 0:
      #run pflotran to get its output
      
      #Given p, update material properties
      p_bar = np.zeros(self.problem_size, dtype='f8')
      for mat_prop in self.mat_props:
        X = mat_prop.convert_p_to_mat_properties(p_bar)
        self.solver.create_cell_indexed_dataset(X, mat_prop.get_name().lower(),
                      X_ids=mat_prop.get_cell_ids_to_parametrize(), resize_to=True)
      #run PFLOTRAN
      if not self.solver.mesh_info_present:
        ret_code = self.solver.run()
        if ret_code: exit(ret_code)
    
    self.filter_i = [np.zeros(self.solver.n_cells, dtype='f8') for x in range(n_inputs)]
    for i,var in enumerate(self.filter.__get_PFLOTRAN_output_variable_needed__()):
      self.solver.get_output_variable(var, self.filter_i[i], -1) #last timestep
    self.filter.set_inputs(self.filter_i)
    self.filter.initialize()
    return
    
  
  def __print_information__(self):
    print("""\n
\t===================================
\t
\t            HydrOpTop
\t
\t   Topology optimization tool for
\t      hydrogeological problem
\t
\t               by
\t       Moise Rousseau (2021)
\t
\t===================================
    """)
    

