import h5py
import numpy as np
import nlopt
import functools

from HydrOpTop.Adjoints import *
from HydrOpTop.IO import IO


class Steady_State_Crafter:
  """
  Craft a topology optimization problem in steady state
  Argument:
  - mat_props: a list of material properties that vary with the density
               parameter p (Material classes instances)
  - solver: object that manage the PDE solver (Solver class instance)
  - objective: the objective function (Function class instance)
  - constraints: a list of constraints (Function class instances
  - filter: the filter to be used to relate the density to a filtered density
            (a fitler class instance) (None by default).
  """
  def __init__(self, objective, solver, mat_props, constraints=[], filters=None, io=None):
    self.__print_information__()
    self.mat_props = mat_props
    self.solver = solver
    self.obj = objective
    self.constraints = constraints
    self.filters = filters
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
    self.iteration = 0
    self.last_p = None
    self.last_grad = None
    self.last_p_bar = None
    self.first_cf = None
    self.first_p = None
    
    self.__initialize_IO_array__()
    self.filters_initialized = False
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
  
  
  def initialize_function_vars(self, func):
    func_var = []
    for var in func.__get_solved_variables_needed__():
      func_var.append(self.solver.get_output_variable(var))
    for var in func.__get_input_variables_needed__():
      func_var.append(self.solver.get_output_variable(var))
    func.set_inputs(func_var)
    return func_var
    
  def update_function_vars(self, func, func_var):
    for i,var in enumerate(func.__get_solved_variables_needed__()):
      self.solver.get_output_variable(var, func_var[i], -1) #last timestep
    i = len(func.__get_solved_variables_needed__())
    for j,var in enumerate(func.__get_input_variables_needed__()):
      self.solver.get_output_variable(var, func_var[i+j], -1) #last timestep
    return
    
  def filter_density(self, p):
    if not self.filters:
      return p
    if not self.filters_initialized:
      self.__initialize_filter__(p)
    p_bar = p.copy()
    for filter_ in self.filters: #apply filter consecutively
      self.initialize_function_vars(filter_) #update filter var
      p_bar = filter_.get_filtered_density(p_bar)
    return p_bar
    
  def evaluate_total_gradient(self, func, p, p_bar=None):
    if self.filters:
      if p_bar is None:
        p_bar = self.filter_density(p)
      p_ = p_bar
    else:
      p_ = p
    dobj_dY = func.d_objective_dY(p_)
    dobj_dp_partial = func.d_objective_dp_partial(p_)
    dobj_dX = func.d_objective_dX(p_)
    grad = func.adjoint.compute_sensitivity(p_, dobj_dY, 
                dobj_dX, func.__get_input_variables_needed__()) + \
                dobj_dp_partial
    if self.filters:
      p_ = p
      for i,filter_ in enumerate(self.filters):
        if not i:
          grad_filter = filter_.get_filter_derivative(p_)
        else:
          grad_filter = filter_.get_filter_derivative(p_).dot(grad_filter)
        p_ = filter_.get_filtered_density(p_)
      grad[:] = grad_filter.transpose().dot(grad)
    return grad
  
  
  def output_to_user(self):
    self.IO.output(self.func_eval, 
                   self.last_cf, 
                   self.last_constraints, 
                   self.last_p, 
                   self.last_grad, 
                   self.last_grad_constraints, 
                   {x.get_name():x.convert_p_to_mat_properties(self.last_p_bar) for x in self.mat_props},
                   self.last_p_bar,
                   val_at=self.p_ids-1)
    return
    
  
  ### PRE-EVALUATION ###
  def pre_evaluation_objective(self, p):
    """
    Perform filtering and projection, parametrization, run the solver 
    and get result (i.e. update function parameter)
    """
    ### FILTERING
    p_bar = self.filter_density(p)
    ### UPDATE MAT PROPERTIES
    for mat_prop in self.mat_props:
      X = mat_prop.convert_p_to_mat_properties(p_bar)
      self.solver.create_cell_indexed_dataset(X, mat_prop.get_name().lower(),
                    X_ids=mat_prop.get_cell_ids_to_parametrize(), resize_to=True)
    ### RUN SOLVER
    ret_code = self.solver.run()
    if ret_code: 
      exit(ret_code)
    ### UPDATE FUNCTION VARIABLES ###
    # Objective
    if self.Yi is None:
      self.Yi = self.initialize_function_vars(self.obj)
    else:
      self.update_function_vars(self.obj, self.Yi)
    # constraints
    if self.constraint_inputs_arrays is None: #need to initialize
      self.constraint_inputs_arrays = []
      for constraint in self.constraints:
        temp = self.initialize_function_vars(constraint[0])
        self.constraint_inputs_arrays.append(temp)
    else:
      for i,constraint in enumerate(self.constraints):
        self.update_function_vars(constraint[0], self.constraint_inputs_arrays[i])
    return p_bar
  
  
  ### OPTIMIZER ###
  def optimize(self, optimizer="nlopt-mma", 
                     initial_guess=None,
                     action="minimize", 
                     density_parameter_bounds=[0.001, 1],
                     tolerance_constraints=0.005,
                     max_it=50,
                     ftol=None,
                     options={}):
    self.first_cf = None
    ### DEFAULT INPUTS
    if initial_guess is None:
      initial_guess = np.zeros(self.get_problem_size(), dtype='f8') + \
                         density_parameter_bounds[0]
    self.first_p = initial_guess.copy()
    ### OPTIMIZE
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
      for i,tc in enumerate(self.constraints):
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
      #optimize
      try:
        p_opt = opt.optimize(initial_guess)
      except(KeyboardInterrupt):
        p_opt = self.last_p
    
    
    elif optimizer in ["scipy-SLSQP", "scipy-trust-constr"]:
      from scipy.optimize import minimize, NonlinearConstraint
      algo = optimizer[6:]
      options["maxiter"] = max_it
      consts = []
      for constraint in self.constraints:
        const = NonlinearConstraint(lambda p: self.scipy_constraint_val(constraint,p),
                                    -np.inf, 0,
                                    jac=lambda p: self.scipy_constraint_jac(constraint,p))
        consts.append(const)
      p_opt = minimize(self.scipy_function_to_optimize, 
                       method=algo,
                       jac=self.scipy_jac, 
                       x0=initial_guess, 
                       bounds=np.repeat([density_parameter_bounds],len(initial_guess),axis=0),
                       constraints=consts, 
                       tol=ftol,
                       options=options,
                       callback=self.scipy_callback)
      
    
    elif optimizer == "cyipopt":
      import cyipopt
      #define problem class
      class Pb:
        def __init__(self,this):
          self.this = this
        def objective(self, p):
          return self.this.scipy_function_to_optimize(p)
        def gradient(self, p):
          return self.this.scipy_jac(p)
        def constraints(self, p):
          vals = [self.this.scipy_constraint_val(x,p) for x in self.this.constraints]
          return np.array(vals)
        def jacobian(self, p):
          grads = [self.this.scipy_constraint_jac(x,p) for x in self.this.constraints]
          return np.concatenate(grads)
        def intermediate(self, alg_mod, iter_count, obj_value, inf_pr, inf_du, mu,
                         d_norm, regularization_size, alpha_du, alpha_pr,
                         ls_trials):
          print(f"Current {self.this.obj.name} (cost function): {obj_value:.6e}")
          for name,val in self.this.last_constraints.items():
            print(f"Current {name} (constraint): {val:.6e}")
          print(f"End of iteration {iter_count}")
          self.this.output_to_user()
          print("-------------------------------------\n")
          return
      
      nlp = cyipopt.Problem(n=len(initial_guess),
                    m=len(self.constraints),
                    problem_obj=Pb(self),
                    lb=[density_parameter_bounds[0] for x in initial_guess],
                    ub=[density_parameter_bounds[1] for x in initial_guess],
                    cl=[0 for x in self.constraints],
                    cu=[1e40 for x in self.constraints])
      nlp.add_option("tol", ftol)
      nlp.add_option("max_iter", max_it)
      nlp.add_option("print_level", 0)
      for key,val in options.items():
        nlp.add_option(key, val)
      p_opt, info = nlp.solve(initial_guess)
    
    else:
      print(f"Error: Unknown optimizer or unadapted \"{optimizer}\"")
      return None
      
    #print output
    #self.output_to_user()
    print("END!")
    
    out = Output_Struct(p_opt)
    if self.filters: 
      out.p_opt_filtered = self.filter_density(p_opt)
    out.fx = self.last_cf
    out.cx = self.last_constraints
    return out
    
  
  ######################
  ### NLOPT FUNCTION ###
  ######################
  def nlopt_function_to_optimize(self, p, grad):
    #save last iteration
    if self.func_eval != 0:
      self.output_to_user()
    #start new it
    self.func_eval += 1
    print(f"\nFonction evaluation {self.func_eval}")
    ### PRE-EVALUATION
    p_bar = self.pre_evaluation_objective(p)
    ### OBJECTIVE EVALUATION AND DERIVATIVE
    cf = self.obj.evaluate(p_bar)
    if self.first_cf is None: 
      self.first_cf = cf
    if grad.size > 0:
      grad[:] = self.evaluate_total_gradient(self.obj, p, p_bar)
    #save for output at next iteration
    self.last_cf = cf
    self.last_grad[:] = grad
    self.last_p[:] = p
    if self.filters:
      self.last_p_bar = p_bar
    #normalize cf to 1
    cf /= self.first_cf
    grad /= self.first_cf
    #print to user
    print(f"Current {self.obj.name} (cost function): {cf*self.first_cf:.6e}")
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
    p_bar = self.filter_density(p)
    the_constraint,compare,val = self.constraints[iconstraint]
    constraint = the_constraint.evaluate(p_bar) - val
    c_name = self.constraints[iconstraint][0].name
    print(f"Current {c_name} (constrain): {constraint+val:.6e}")
    if grad.size > 0:
      grad[:] = self.evaluate_total_gradient(the_constraint, p, p_bar)
    if compare == '>':
      grad *= -1.
      constraint = -constraint
    
    self.last_constraints[c_name] = constraint
    self.last_grad_constraints[c_name] = grad.copy()
    return constraint - the_constraint.__get_constraint_tol__()
    
    
  ########################
  ###  SCIPY FUNCTIONS ###
  ########################
  def scipy_run_sim(self, p):
    if not np.allclose(p, self.last_p):
      self.func_eval += 1
      self.last_p_bar[:] = self.pre_evaluation_objective(p)
      self.last_p[:] = p
    return
  
  def scipy_function_to_optimize(self,p):
    self.scipy_run_sim(p)
    cf = self.obj.evaluate(self.last_p_bar)
    self.last_cf = cf  
    return cf
  
  def scipy_jac(self,p):
    self.scipy_run_sim(p)
    grad = self.evaluate_total_gradient(self.obj, self.last_p, self.last_p_bar)
    self.last_grad[:] = grad
    return grad
  
  def scipy_constraint_val(self, constraint, p):
    self.scipy_run_sim(p)
    val = constraint.evaluate(self.last_p_bar)
    self.last_constraints[constraint.name] = val
    return -val + constraint.__get_constraint_tol__()
  
  def scipy_constraint_jac(self, constraint, p):
    self.scipy_run_sim(p)
    grad = -self.evaluate_total_gradient(constraint, p, self.last_p_bar) #don't forget the minus
    self.last_grad_constraints[constraint.name] = grad.copy()
    return grad
  
  def scipy_callback(self):
    #self.output_to_user()
    self.iteration += 1
    print(f"\nIteration {self.iteration}")
    print(f"Solver call: {self.func_eval}")
    print(f"Current {self.obj.name} (cost function): {cf:.6e}")
    for name,val in self.last_constraints.items():
      print(f"Current {name} (constraint): {val:.6e}")
    return
  
  def _print_optimization_info_to_user(self):
    return
    
  
  
  ### INITIALIZATION ###
  def __initialize_IO_array__(self):
    print("Initialization...")
    #verify if each cells to parametrize are the same
    X = self.mat_props[0].get_cell_ids_to_parametrize()
    if len(self.mat_props) > 1:
      for x in self.mat_props:
        if x.get_cell_ids_to_parametrize() != X: 
          print("Different cell ids to optimize")
          print("HydrOpTop require the different mat properties to parametrize \
                 the same cell ids")
          exit(1)
    #create correspondance and problem size
    if X is None: #i.e. parametrize all cell in the simulation
      self.problem_size = self.solver.get_grid_size()
      self.p_ids = np.arange(1, self.problem_size+1) 
    else: 
      self.problem_size = len(X)
      self.p_ids = X #from 0 based indexing to solver indexing
    self.last_p = np.zeros(len(self.p_ids),dtype='f8')
    self.last_grad = np.zeros(len(self.p_ids),dtype='f8')
    self.last_p_bar = np.zeros(len(self.p_ids),dtype='f8')
    if self.constraints:
      pass
    
    #initialize solver output for objective function
    #do not set inputs array because we don't know the size of the connection_ids
    #in case of face output
    
    #initialize adjoint for objective function
    #TODO: what happen for 2 variables butlowly or highly coupled ??
    if self.obj.adjoint is None: #could have been set by user
      if len(self.obj.__get_solved_variables_needed__()) == 0: #no adjoint required
        adjoint = No_Adjoint(self.mat_props, self.p_ids)
      elif len(self.obj.__get_solved_variables_needed__()) == 1: #one variable
        adjoint = Sensitivity_Steady_Simple(self.obj.__get_solved_variables_needed__(),
                                       self.mat_props, self.solver, self.p_ids)
      self.obj.set_adjoint_problem(adjoint)
      
    self.obj.set_p_to_cell_ids(self.p_ids)
    
    #initialize adjoint for constraints
    self.last_constraints = {x[0].name:0. for x in self.constraints}
    self.last_grad_constraints = {x[0].name:None for x in self.constraints}
    for constraint in self.constraints:
      constraint = constraint[0]
      constraint.set_p_to_cell_ids(self.p_ids)
      if constraint.adjoint is None:
        if len(constraint.__get_solved_variables_needed__()) == 0:
          adjoint = No_Adjoint(self.mat_props, self.p_ids)
        elif len(constraint.__get_solved_variables_needed__()) == 1:
          adjoint = Sensitivity_Steady_Simple(constraint.__get_solved_variables_needed__(),
                                             self.mat_props, self.solver, self.p_ids)
        constraint.set_adjoint_problem(adjoint)
        
    #initialize IO
    self.IO.communicate_functions_names(self.obj.__get_name__(), 
         [x[0].__get_name__() for x in self.constraints])
    vertices, cells, indexes = self.solver.get_mesh()
    self.IO.set_mesh_info(vertices, cells, indexes,
                          self.solver.get_var_location())
    return
    
  
  def __initialize_filter__(self, p):
    print("Filter initialization")
    #filter initialization is tricky, because it may need solver output variables
    # that have not been created yet. Thus, for instance, we have to run
    # solver one time to initialize the filter (even if it's costly)...
    if self.filters is None:
      return
    
    for mat_prop in self.mat_props:
      X = mat_prop.convert_p_to_mat_properties(p)
      self.solver.create_cell_indexed_dataset(X, mat_prop.get_name().lower(),
                      X_ids=mat_prop.get_cell_ids_to_parametrize(), resize_to=True)
    #run Solver
    for filter_ in self.filters:
      if filter_.__get_input_variables_needed__() or filter_.__get_solved_variables_needed__():
        ret_code = self.solver.run()
        if ret_code: 
          exit(ret_code)
    
    for filter_ in self.filters:
      self.initialize_function_vars(filter_)
      filter_.set_p_to_cell_ids(self.p_ids)
    self.filters_initialized = True
    return
    
  
  def __print_information__(self):
    print("""\n
\t===================================
\t
\t             HydrOpTop
\t
\t   Topology optimization using a 
\t       modular, flexible and 
\t    solver-independent approach
\t
\t               by
\t       Moise Rousseau (2022)
\t
\t===================================
    """)


class Output_Struct:
  def __init__(self, p_opt):
    self.p_opt = p_opt
    self.p_opt_filtered = None
    self.p_opt_grad = None
    self.fx = None
    self.cx = None
    return
  
  def __repr__(self):
    out = "<HydrOpTop results:\n"
    out += f"p_opt: {self.p_opt}\n"
    if self.p_opt_filtered is not None:
      out += f"p_opt_filtered: {self.p_opt_filtered}\n"
    out += f"fx: {self.fx}\ncx: {self.cx}>"
    return out
    
    

