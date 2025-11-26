import numpy as np
import scipy.sparse as sp
import nlopt
import functools
import time

from HydrOpTop.Adjoints import *
from HydrOpTop.IO import IO
from HydrOpTop.Filters import Pilot_Points


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
  - adjoint: a linear solver object that solve the adjoint equation
  """
  def __init__(self, objective, solver, mat_props, constraints=[], filters=[], deriv=None, deriv_args=None, io=None):
    self.__print_information__()
    self.mat_props = mat_props
    self.solver = solver
    self.obj = objective
    self.constraints = constraints
    self.filters = filters
    self.adjoint = deriv
    self.adjoint_args = deriv_args
    if io is None:
      self.IO = IO()
    else:
      self.IO = io
    
    #self.Xi = None #store material properties (solver inputs)
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
    self.best_p = (None,None) #best cf, best p
    self.last_p = None
    self.last_grad = None
    self.last_p_bar = None
    self.first_p = None

    if self.adjoint is None:
      pass
    
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
  
  def update_function_vars(self):
    # Cache the solver var not to open output file repetively
    all_var_needed = set(self.obj.variables_needed)
    for c in self.constraints:
      all_var_needed = all_var_needed.union(c[0].variables_needed)
    func_var = {var:None for var in all_var_needed}
    self.solver.get_output_variables(func_var)
    self.obj.set_inputs({k:v[self.obj.indexes] for k,v in func_var.items() if k in self.obj.variables_needed})
    for c in self.constraints:
      c[0].set_inputs({k:v[c[0].indexes] for k,v in func_var.items() if k in c[0].variables_needed})
    return func_var

  def filter_density(self, p):
    if not self.filters:
      return p
    # p is a vector of input dim, with both reduced dim
    p_bar = np.zeros_like(self.last_p_bar)
    p_bar[self.input_cells] = p
    for filter_ in self.filters: #apply filter consecutively
      # Filter does not need to be updated
      #self.update_function_vars() #update filter var
      p_bar[filter_.output_indexes] = filter_.get_filtered_density(p_bar[filter_.input_indexes])
    return p_bar
  
  def evaluate_objective(self, p):
    """
    Evaluate objective
    """
    if not isinstance(self.obj.indexes, slice):
      ids_to_p_index = self.ids_p[self.obj.indexes+self.solver.cell_id_start_at]
    else:
      ids_to_p_index = self.ids_p[self.obj.indexes]
    p_ = np.where(ids_to_p_index == -1, np.nan, p[ids_to_p_index])
    cf = self.obj.evaluate(p_)
    self.last_cf = cf
    if np.any(np.isnan(cf)):
      raise RuntimeError("Objective function is NaN. Check if you defined it where some variable are not defined")
    return cf

  def evaluate_total_gradient(self, func, p, p_bar=None, cf=None):
    tstart = time.time()

    if not np.allclose(p, self.last_p):
      self.func_eval += 1
      self.last_p[:] = p
      self.last_p_bar[:] = self.pre_evaluation_objective(p)
      self.last_cf = self.evaluate_objective(self.last_p_bar)

    if isinstance(func.adjoint, Sensitivity_Steady_Simple) or isinstance(func.adjoint, No_Adjoint):
      if self.filters:
        if p_bar is None:
          p_bar = self.filter_density(p)
        p_ = p_bar
      else:
        p_ = p
      # derivative according to solved var
      dobj_dY = {}
      dobj_dX = {}
      # p for function
      if not isinstance(func.indexes, slice):
        ids_to_p_index = self.ids_p[func.indexes+self.solver.cell_id_start_at]
        pf = np.where(ids_to_p_index == -1, np.nan, p_[ids_to_p_index])
      else:
        ids_to_p_index = self.ids_p[func.indexes]
        pf = np.where(ids_to_p_index == -1, np.nan, p_[ids_to_p_index])[self.solver.cell_id_start_at:]
      for var in func.__get_variables_needed__():
        var_ = var.replace("_INTERPOLATOR","")
        if var_ in self.solver.solved_variables:
          if isinstance(self.last_cf, float):
            dobj_dY[var_] = np.zeros(self.solver.get_system_size())
          else:
            dobj_dY[var_] = np.zeros((self.solver.get_system_size(),len(self.last_cf)))
          dobj_dY[var_][func.indexes] = func.d_objective(var, pf)
        elif var_ in [x.get_name() for x in self.mat_props]:
          dobj_dX[var_] = np.zeros(self.solver.get_grid_size())
          dobj_dX[var_][func.indexes] = func.d_objective(var, pf)
      dobj_dp_partial = np.zeros(self.solver.get_grid_size())
      dobj_dp_partial[func.indexes] = func.d_objective_dp_partial(pf)

      grad = func.adjoint.compute_sensitivity(
        p_, dobj_dY, dobj_dX,
      )
      if isinstance(grad, np.ndarray) and grad.ndim == 2:
        grad += dobj_dp_partial[self.p_ids-self.solver.cell_id_start_at,None]
      else:
        grad += dobj_dp_partial[self.p_ids-self.solver.cell_id_start_at]

      if self.filters:
        # Compute d_p_bar / d_p
        p_bar = np.zeros_like(self.last_p_bar)
        p_bar[self.input_cells] = p
        N = len(p_bar)
        Jf = sp.diags(np.ones(N),format="csr")
        for i,f in enumerate(self.filters):
          # We are in simulation space here
          # and filter acts as a bijection, we take the column we want at the end
          Jp = f.get_filter_derivative(p_bar[f.input_indexes])
          global_row = f.output_indexes[Jp.coords[0]]
          global_col = f.input_indexes[Jp.coords[1]]
          # add 1 to indexes unchanged by the filter
          uc = np.nonzero(~np.isin(np.arange(0,N),f.output_indexes))[0]
          global_row = np.concat([global_row,uc])
          global_col = np.concat([global_col,uc])
          data = np.concat([Jp.data,np.ones(len(uc))])
          J = sp.coo_matrix(
            (data,(global_row,global_col)), shape=(N,N)
          )
          # Update global filter derivative
          Jf = J.dot(Jf)
          # Update filter state
          p_bar[f.output_indexes] = f.get_filtered_density(
            p_[f.input_indexes]
          )
        grad = Jf[:,self.input_cells].transpose().dot(grad)

    elif isinstance(func.adjoint, Sensitivity_Finite_Difference):
      if self.last_cf is not None:
        func.adjoint.set_current_obj_val(self.last_cf)
      grad = func.adjoint.compute_sensitivity(p)

    print("Total time to get derivative:", time.time() - tstart, "s")
    return grad
  
  
  def output_to_user(self, final=False):
    val_at = np.argwhere(np.isin(
      self.solver.get_region_ids("__all__"), self.p_ids
    )).flatten()
    p =  self.best_p[1] if final else self.last_p
    p_bar = self.filter_density(p) if final else self.last_p_bar
    cf = self.last_cf if not isinstance(self.last_cf,np.ndarray) else np.linalg.norm(self.last_cf)
    self.IO.output(
      it=self.iteration, #iteration number
      cf=cf, #cost function value
      constraints_val=self.last_constraints, #constraints value
      p_raw=p, #raw density parameter (not filtered)
      grad_cf=self.last_grad, # d(cost function) / d p
      grad_constraints=self.last_grad_constraints, 
      mat_props=self.get_material_properties_from_p(p_bar),
      p_filtered=p_bar,
      adj_obj=self.obj.adjoint.l0,
      val_at=val_at,
      final=final,
    )
    return


  def get_material_properties_from_p(self, p_):
    """
    Return material properties corresponding to the given value of density parameter
    """
    mat_prop_unique = set([m.get_name() for m in self.mat_props])
    ret = {}
    for mu in mat_prop_unique:
      X = np.zeros_like(p_)
      for mat_prop in self.mat_props:
        if mat_prop.get_name() != mu: continue
        X[mat_prop.indexes] = mat_prop.convert_p_to_mat_properties(p_[mat_prop.indexes])
      ret[mu] = X
    return ret
    
  
  ### PRE-EVALUATION ###
  def pre_evaluation_objective(self, p):
    """
    Perform filtering and projection, parametrization, run the solver 
    and get result (i.e. update function parameter)
    """
    ### FILTERING
    p_bar = self.filter_density(p)
    ### UPDATE MAT PROPERTIES
    dict_mat = self.get_material_properties_from_p(p_bar)
    for name,val in dict_mat.items():
      self.solver.create_cell_indexed_dataset(
        val, name, X_ids=self.p_ids, resize_to=False
      )
    ### RUN SOLVER
    ret_code = self.solver.run()
    if ret_code: 
      exit(ret_code)
    ### UPDATE FUNCTION VARIABLES ###
    self.update_function_vars()
    self.last_p[:] = p
    self.last_p_bar[:] = p_bar
    return p_bar
  
  
  ### OPTIMIZER ###
  def optimize(self, optimizer="nlopt-mma", 
                     initial_guess=None,
                     action="minimize", 
                     density_parameter_bounds=[0.001, 1],
                     tolerance_constraints=0.005,
                     max_it=50,
                     initial_step=None,
                     stop={},
                     options={}):
    self.iteration = 0
    self.func_eval = 0
    ### DEFAULT INPUTS
    if initial_guess is None:
      initial_guess = np.zeros(self.get_problem_size(), dtype='f8') + \
                         np.mean(density_parameter_bounds)
    self.first_p = initial_guess.copy()
    ### OPTIMIZE
    if "nlopt" in optimizer:
      d_algo = {
        "nlopt-mma":nlopt.LD_MMA,
        "nlopt-ccsaq":nlopt.LD_CCSAQ,
        "nlopt-slsqp":nlopt.LD_SLSQP,
        "nlopt-ptn": nlopt.LD_TNEWTON_PRECOND,
        "nlopt-lbfgs": nlopt.LD_LBFGS,
      }
      algorithm = d_algo[optimizer]
      opt = nlopt.opt(algorithm, self.problem_size)
      if action == "minimize":
        opt.set_min_objective(self.nlopt_function_to_optimize) 
      elif action == "maximize":
        opt.set_max_objective(self.nlopt_function_to_optimize) 
      else:
        raise RuntimeError(f"Error: Unknown action \"{action}\" (should be \"minimize\" or \"maximize\"")
      #add constraints
      for i,tc in enumerate(self.constraints):
        opt.add_inequality_constraint(self.nlopt_constraint(i),
                                      tolerance_constraints)
      #define minimum and maximum bounds
      opt.set_lower_bounds(np.zeros(self.get_problem_size(), dtype='f8') +
                           density_parameter_bounds[0])
      opt.set_upper_bounds(np.zeros(self.get_problem_size(), dtype='f8') +
                           density_parameter_bounds[1])
      #opt.set_param("inner_maxeval", 50)
      if initial_step is not None:
        opt.set_initial_step(initial_step)
      #define stop criterion
      opt.set_maxeval(max_it)
      if stop:
        if "ftol_rel" in stop.keys(): opt.set_ftol_rel(stop["ftol_rel"])
        if "ftol_abs" in stop.keys(): opt.set_ftol_abs(stop["ftol_abs"])
        if "stopval" in stop.keys(): opt.set_stopval(stop["stopval"])

      # adjust initial step size
      #opt.set_initial_step(20)
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
      res = minimize(
         self.scipy_function_to_optimize, 
         method=algo,
         jac=self.scipy_jac, 
         x0=initial_guess,
         bounds=np.repeat([density_parameter_bounds],len(initial_guess),axis=0),
         constraints=consts, 
         tol=stop["ftol_rel"],
         options=options,
         callback=self.scipy_callback
      )
      p_opt = res.x
      print(f"Optimizer {algo} exited with success = {res.success}. Reason: {res.message}")
    
    elif optimizer == "scipy-trf" or optimizer == "scipy-dogbox":
      from scipy.optimize import least_squares
      res = least_squares(
        self.scipy_function_to_optimize,
        x0=initial_guess,
        jac=self.scipy_jac,
        bounds=np.repeat([density_parameter_bounds],len(initial_guess),axis=0).T,
        ftol=stop.get("ftol",1e-6),
        method=optimizer[6:],
        #loss="soft_l1",
        max_nfev=max_it,
        callback=self.scipy_callback,
        verbose=2,
      )
      p_opt = res.x
      self.best_p = (res.cost, res.x)
      self.last_cf = np.sqrt(np.mean(res.fun**2))
    
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
      if "ftol" in stop.keys():
        nlp.add_option("tol", stop["ftol"])
      nlp.add_option("max_iter", max_it)
      nlp.add_option("print_level", 0)
      for key,val in options.items():
        nlp.add_option(key, val)
      p_opt, info = nlp.solve(initial_guess)
    
    else:
      print(f"Error: Unknown optimizer or unadapted \"{optimizer}\"")
      return None
      
    #print last iteration
    print("\nOptimization finished!")
    print(f"Best {self.obj.name} cost function value: {self.best_p[0]}")
    print("Write optimum")
    self.output_to_user(final=True)
    print("END!")
    
    # Review this with best value
    out = Output_Struct(self.best_p[1])
    if self.filters: 
      out.p_opt_filtered = self.filter_density(p_opt)
    out.fx = self.best_p[0]
    out.grad_fx = self.last_grad
    out.cx = self.last_constraints
    out.grad_cx = self.last_grad_constraints
    out.func_eval = self.func_eval
    out.mat_props = self.get_material_properties_from_p(self.last_p_bar)
    return out
    
  
  ######################
  ### NLOPT FUNCTION ###
  ######################
  def nlopt_function_to_optimize(self, p, grad):
    #start new it
    self.func_eval += 1
    self.iteration += 1 #for mma and ccsaq, one func call per iteration
    print(f"\nFonction evaluation {self.func_eval}")
    ### PRE-EVALUATION
    p_bar = self.pre_evaluation_objective(p)
    ### OBJECTIVE EVALUATION AND DERIVATIVE
    cf = self.evaluate_objective(p_bar)
    self.last_cf = cf
    print(f"Current {self.obj.name} (cost function): {cf:.6e}")
    # Derivative
    if grad.size > 0:
      grad[:] = self.evaluate_total_gradient(self.obj, p, p_bar, cf)
      self.last_grad[:] = grad
      print(f"Min gradient: {np.min(grad):.6e} at cell id {np.argmin(grad)+1}")
      print(f"Max gradient: {np.max(grad):.6e} at cell id {np.argmax(grad)+1}")
    self.last_p[:] = p
    if self.filters:
      self.last_p_bar[:] = p_bar
    # Store the best iterate
    if self.best_p[0] is None or cf < self.best_p[0]:
        self.best_p = (cf, p.copy())
    self.output_to_user()
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
    constraint = the_constraint.evaluate(p_bar)
    c_name = self.constraints[iconstraint][0].name
    print(f"Current {c_name} (constrain): {constraint:.6e}")
    if grad.size > 0:
      grad[:] = self.evaluate_total_gradient(the_constraint, p, p_bar)
    if compare == '>':
      if grad.size > 0: grad[:] *= -1.
      constraint = -constraint
      val *= -1
    
    self.last_constraints[c_name] = constraint
    self.last_grad_constraints[c_name] = grad.copy()
    return constraint - val
    
    
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
    print("\nNew function evaluation")
    self.scipy_run_sim(p)
    cf = self.evaluate_objective(self.last_p_bar)
    self.last_cf = cf
    if self.best_p[0] is None or np.linalg.norm(cf) < np.linalg.norm(self.best_p[0]):
        self.best_p = (cf, p.copy())
    return cf
  
  def scipy_jac(self,p):
    self.scipy_run_sim(p)
    grad = self.evaluate_total_gradient(self.obj, self.last_p, self.last_p_bar)
    self.last_grad = grad.T
    return grad.T
  
  def scipy_constraint_val(self, constraint, p):
    self.scipy_run_sim(p)
    val = constraint[0].evaluate(self.last_p_bar)
    self.last_constraints[constraint[0].name] = val
    if constraint[1] == '<':
      return -val + constraint[2]
    elif constraint[1] == '>':
      return val - constraint[2]
  
  def scipy_constraint_jac(self, constraint, p):
    self.scipy_run_sim(p)
    grad = self.evaluate_total_gradient(constraint[0], p, self.last_p_bar) #don't forget the minus
    self.last_grad_constraints[constraint[0].name] = grad.copy()
    return grad
  
  def scipy_callback(self, x, res=None):
    self.iteration += 1
    self.output_to_user()
    return
  
  def _print_optimization_info_to_user(self):
    return
    
  
  
  ### INITIALIZATION ###
  def __initialize_IO_array__(self):
    print("Initialization...")
    # Get all the cell which are parametrized
    parameterized_cells = set()
    for x in self.mat_props:
      cids = x.get_cell_ids_to_parametrize()
      if cids is None: # Mean all cells
        parameterized_cells = set(self.solver.get_region_ids("__all__"))
        break
      parameterized_cells = parameterized_cells.union(cids)
    self.p_ids = np.array(list(parameterized_cells)) # Correspondance between p and id in simulation
    self.ids_p = -np.ones(self.solver.get_grid_size()+1, dtype='i8') #+1 because it can be 0 or 1 based
    self.ids_p[self.p_ids] = np.arange(len(self.p_ids)) #now we can query self.ids_p[cell_ids] and get index in p (or -1 if not attributed)
  
    self.p_bar = np.zeros(len(self.p_ids))
    # Assign which mat prop should work on which p indexes
    for m in self.mat_props:
      m.indexes = np.nonzero(np.isin(self.p_ids, m.cell_ids))[0]
    
    # Filter initialisation
    # During filtering, the filter will received an array of size len(input_ids) and an array len(output_ids) to populate
    # At the end, the output array is the filtered p with correspondance p to cell ids (self.p_ids)
    # Here we determine which filter write where in this output array
    input_cell = set()
    output_cell = set()
    for f in self.filters:
      if f.input_ids is None:
        input_cell = set(self.solver.get_region_ids("__all__"))
        break
      else:
        input_cell = input_cell.union([x for x in f.input_ids if x >= 0])
      if f.output_ids is None:
        output_cell = set(self.solver.get_region_ids("__all__"))
        break
      else:
        output_cell = output_cell.union(f.output_ids)
    not_filtered_cell = parameterized_cells.difference(output_cell)
    input_dim = len(not_filtered_cell) + len(input_cell)
    # Now we have input dimension, let tell to the filter where to write
    for f in self.filters:
      f.input_indexes = np.nonzero(np.isin(self.p_ids, f.input_ids))[0]
      f.output_indexes = np.nonzero(np.isin(self.p_ids, f.output_ids))[0]
    # Now store correspondance between input parameter p and the location to write in p_bar
    self.input_cells = np.nonzero(np.isin(
      self.p_ids, list(input_cell.union(not_filtered_cell))
    ))[0]
    
    #create correspondance and problem size
    self.problem_size = input_dim
    #self.p = np.zeros(input_dim, dtype='f8')
    self.last_p = np.zeros(input_dim,dtype='f8')
    self.last_grad = np.zeros(input_dim,dtype='f8')
    self.last_p_bar = np.zeros_like(self.p_ids,dtype='f8')
    if self.constraints:
      pass
    
    #initialize solver output for objective function
    #do not set inputs array because we don't know the size of the connection_ids
    #in case of face output
    
    #initialize adjoint for objective function
    #TODO: what happen for 2 variables but lowly or highly coupled ??
    #TODO: adjoint should not be an attribute of the objective function
    if self.adjoint is None:
      self.adjoint = "adjoint"
    if self.obj.adjoint is None: #could have been set by user
      solved_variables_needed = []
      for var in self.obj.__get_variables_needed__():
        var_ = var.replace("_INTERPOLATOR","")
        if var_ in self.solver.solved_variables:
          solved_variables_needed.append(var_)
      if len(solved_variables_needed) == 0:
        adjoint = No_Adjoint(self.mat_props, self.p_ids)
      elif self.adjoint == "fd":
        f = lambda p: self.evaluate_objective(self.pre_evaluation_objective(p))
        adjoint = Sensitivity_Finite_Difference(f, **self.adjoint_args)
      elif self.adjoint == "adjoint":
        adjoint = Sensitivity_Steady_Simple(
          solved_variables_needed,
          self.mat_props,
          self.solver,
          self.p_ids-self.solver.cell_id_start_at,
          *( (self.adjoint_args,) if self.adjoint_args is not None else () )
        )
      self.obj.set_adjoint_problem(adjoint)
    
    if self.obj.indexes is None:
      # pass all cell
      self.obj.indexes = slice(None)
    else:
      self.obj.indexes -= self.solver.cell_id_start_at
      
    self.obj.set_p_to_cell_ids(self.p_ids)
    
    #initialize adjoint for constraints
    self.last_constraints = {x[0].name:0. for x in self.constraints}
    self.last_grad_constraints = {x[0].name:None for x in self.constraints}
    for constraint in self.constraints:
      constraint = constraint[0]
      #constraint.set_p_to_cell_ids(self.p_ids)
      # set adjoint
      if constraint.adjoint is None:
        solved_variables_needed = []
        for var in constraint.__get_variables_needed__():
          var_ = var.replace("_INTERPOLATOR","")
          if var_ in self.solver.solved_variables:
            solved_variables_needed.append(var_)
        if len(solved_variables_needed) == 0:
          adjoint = No_Adjoint(self.mat_props, self.p_ids)
        elif self.adjoint == "fd":
          f = lambda p: self.evaluate_objective(self.pre_evaluation_objective(p))
          adjoint = Sensitivity_Finite_Difference(f)
        else:
          adjoint = Sensitivity_Steady_Simple(
            solved_variables_needed,
            self.mat_props, self.solver, self.p_ids
          )
        constraint.set_adjoint_problem(adjoint)
        # set indexes
        if constraint.indexes is None: # pass all cell
          constraint.indexes = slice(None)
        else:
          constraint.indexes -= self.solver.cell_id_start_at
        
    #initialize IO
    self.IO.communicate_functions_names(self.obj.__get_name__(), 
         [x[0].__get_name__() for x in self.constraints])
     #iteration number
     #cf, #cost function value
     #constraints_val, #constraints value
     #p_raw, #raw density parameter (not filtered)
     #grad_cf, # d(cost function) / d p
     #grad_constraints, # d(constraints) / d p
     #mat_props, # parametrized mat props (dict)
     #p_filtered, #filtered density parameters
     #val_at=None): # cell/node ids corresponding to dataset
    var_loc = {
      "Density parameter":"cell",
      "Density parameter filtered":"cell",
    }
    var_loc[f"Gradient d{self.obj.__get_name__()}_dp"] = var_loc["Density parameter"]
    var_loc.update({
      x[0].__get_name__(): self.solver.get_var_location(
        x[0].__get_variables_needed__()[0]
      ) for x in self.constraints
    })
    var_loc.update({
      x.get_name(): self.solver.get_var_location(
        x.get_name()
      ) for x in self.mat_props
    })
    var_loc["Adjoint vector"] = "cell" #TODO must be solver attribute
    self.IO.communicate_var_location(var_loc)
    vertices, cells, indexes = self.solver.get_mesh()
    self.IO.set_mesh_info(vertices, cells, indexes)
    return
    
  
  def __initialize_filter__(self):
    if not self.filters:
      return
    # pass solver variable to filter
    all_var_needed = set()
    for f in self.filters:
      all_var_needed = all_var_needed.union(f.variables_needed)
    func_var = {var:None for var in all_var_needed}
    self.solver.get_output_variables(func_var)
    for f in self.filters:
      f.set_inputs({k:v[f.output_ids-1] for k,v in func_var.items() if k in f.variables_needed})
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
\t       Moise Rousseau (2025)
\t
\t===================================
    """)


class Output_Struct:
  def __init__(self, p_opt):
    self.p_opt = p_opt #raw density parameter (not filtered)
    self.p_opt_filtered = None
    self.p_opt_grad = None # d(cost function) / d p
    self.fx = None #cost function value
    self.grad_fx = None
    self.cx = None #constraints value
    self.grad_cx = None
    self.func_eval = 0 #iteration number
    self.mat_props = {}
    return
  
  def __repr__(self):
    out = "<HydrOpTop results:\n"
    out += f"p_opt: {self.p_opt}\n"
    if self.p_opt_filtered is not None:
      out += f"p_opt_filtered: {self.p_opt_filtered}\n"
    out += f"fx: {self.fx}\ncx: {self.cx}>"
    return out
    
    

