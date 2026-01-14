import numpy as np
import scipy.sparse as sp
import nlopt
import functools
import time
import datetime, os

from HydrOpTop.Adjoints import *
from HydrOpTop.IO import IO
from HydrOpTop.Filters import Unit_Filter, Filter_Sequence

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
  def __init__(self, objective, solver, mat_props, constraints=[], filters=[], deriv=None, deriv_args={}, io=None):
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
    self.func_eval_history = []
    self.iteration = 0
    self.best_p = (None,None) #best cf, best p
    self.last_cf = None
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
    self.obj.set_inputs({k:v[self.obj.indexes-self.solver.cell_id_start_at] for k,v in func_var.items() if k in self.obj.variables_needed})
    for c in self.constraints:
      c[0].set_inputs({k:v[c[0].indexes-self.solver.cell_id_start_at] for k,v in func_var.items() if k in c[0].variables_needed})
    return func_var

  def evaluate_objective(self, p_bar):
    """
    Evaluate objective
    """
    # the p below must be in solver indexing
    # we set it to NaN
    p_ = np.zeros( max(self.obj.indexes.max(),self.p_ids.max())+1 ) + np.nan
    p_[self.p_ids] = p_bar
    p_ = p_[self.obj.indexes]
    cf = self.obj.evaluate(p_)
    self.last_cf = cf
    if np.any(np.isnan(cf)):
      raise RuntimeError("Objective function is NaN. Check if you defined it where some variable are not defined")
    return cf

  def evaluate_total_gradient(self, func, p, p_bar=None, cf=None):
    tstart = time.time()

    if not np.allclose(p, self.last_p):
      self.func_eval += 1
      p_bar = self.pre_evaluation_objective(p)
      self.last_cf = self.evaluate_objective(p_bar)

    if isinstance(func.adjoint, Sensitivity_Steady_Simple) or isinstance(func.adjoint, No_Adjoint):
      grad = func.adjoint.compute_sensitivity(
        func, self.filter_sequence, p
      )

    elif (
      isinstance(func.adjoint, Sensitivity_Finite_Difference) or
      isinstance(func.adjoint, Sensitivity_Steady_Adjoint_Corrected) or
      isinstance(func.adjoint, Sensitivity_Ensemble)
    ):
      if self.last_cf is not None:
        func.adjoint.set_current_obj_val(self.last_cf)
      grad, feval = func.adjoint.compute_sensitivity(func, self.filter_sequence, p)
      self.func_eval += feval

    print("Total time to get derivative:", time.time() - tstart, "s")
    return grad
  
  
  def output_to_user(self, final=False):
    if not self.func_eval: return #first call for nlopt
    val_at = self.p_ids-self.solver.cell_id_start_at
    p =  self.best_p[1] if final else self.last_p
    p_bar = self.filter_sequence.filter(p) if final else self.last_p_bar
    cf = self.last_cf if not isinstance(self.last_cf,np.ndarray) else np.linalg.norm(self.last_cf)
    func_var = {var:None for var in self.IO.sim_extra_var}
    sim_extra_var = self.solver.get_output_variables(func_var)
    sim_extra_var_loc = {x:self.solver.get_var_location(x) for x in self.IO.sim_extra_var}
    sim_extra_var = {var:(sim_extra_var_loc[var],sim_extra_var[var]) for var in sim_extra_var.keys()}
    obj_extra_var = self.obj.output_to_user() if self.IO.obj_extra_var else {}
    for var,(loc,ids,X) in obj_extra_var.items():
      obj_extra_var[var] = (loc,ids - self.solver.cell_id_start_at,X)
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
      sim_extra_var=sim_extra_var,
      obj_extra_var=obj_extra_var,
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
        indexes = self.ids_p[mat_prop.cell_ids]
        X[indexes] = mat_prop.convert_p_to_mat_properties(p_[indexes])
      ret[mu] = X
    return ret
    
  
  ### PRE-EVALUATION ###
  def pre_evaluation_objective(self, p):
    """
    Perform filtering and projection, parametrization, run the solver 
    and get result (i.e. update function parameter)
    """
    ### FILTERING
    p_bar = self.filter_sequence.filter(p)
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
                     **options):
    self.iteration = 0
    self.func_eval = 0
    self.action = action
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

      self.output_to_user(final=True)
    
    
    elif optimizer.lower() in ["scipy-slsqp", "scipy-trust-constr"]:
      from scipy.optimize import minimize, NonlinearConstraint
      algo = optimizer[6:]
      options["maxiter"] = max_it
      consts = []
      for constraint in self.constraints:
        const = NonlinearConstraint(
          lambda p: self.scipy_constraint_val(constraint,p),
          0, np.inf,
          jac=lambda p: self.scipy_constraint_jac(constraint,p),
          finite_diff_rel_step=1e-4,
          keep_feasible=True,
        )
        consts.append(const)
      pre = -1. if action == "maximize" else 1.
      res = minimize(
         lambda x: pre * self.scipy_function_to_optimize(x),
         method=algo,
         jac= lambda x: pre * self.scipy_jac(x),
         hessp=(lambda x,p: np.zeros_like(p)) if self.obj.linear else None,
         x0=initial_guess,
         bounds=np.repeat([density_parameter_bounds],len(initial_guess),axis=0),
         constraints=consts, 
         options=options,
         callback=self.scipy_callback,
         **stop,
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
        method=optimizer[6:],
        #loss="soft_l1",
        max_nfev=max_it,
        callback=self.scipy_callback,
        verbose=2,
        **stop
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

    elif optimizer == "PEST-Wrapper":
      from .PEST_Wrapper import PEST_Wrapper
      from ..Functions.least_square_calibration import Least_Square_Calibration
      assert isinstance(self.obj, Least_Square_Calibration), "PEST optimizer only available for Least_Square_Calibration function"
      # create a temporary rundir
      rundir = "PEST_run_" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
      os.mkdir(rundir)
      # Copy model file in the temp dir
      self.solver.copy_model(rundir)
      # Create PEST
      use_jac = options.pop("use_jac") if "use_jac" in options.keys() else False
      jac = self.scipy_jac if use_jac else None
      solver = PEST_Wrapper(
        NPAR=len(initial_guess), NOBS=len(self.obj.ref_head),
        NOPTMAX=max_it,
        RELPARSTP=stop["xtol"],
        rundir=rundir,
        **options,
      )
      res = solver.fit(
        self.scipy_function_to_optimize, x0=initial_guess,
        bounds=np.repeat([density_parameter_bounds],len(initial_guess),axis=0),
        jac=jac,
      )
      if res is None: return #mean we just have created the pest model file
      p_opt = res["x"]
      self.func_eval_history = res["func_eval_history"]

    elif optimizer == "pestlike-lm":
      from .PESTMarquardtLS import PESTMarquardtLS
      solver = PESTMarquardtLS(
        #lambda_init=100,
        #lambda_factors=[0.07, 1.2, 0.7, 1.2, 70, 120],
        maxiter=max_it,
        callback=self.scipy_callback,
        **stop,
        verbose=True
      )
      res = solver.fit(
        self.scipy_function_to_optimize, x0=initial_guess,
        bounds=density_parameter_bounds, jac=self.scipy_jac,
      )
      p_opt = res["x"]
      print(res)

    elif optimizer == "spsa":
      from .spsa import SPSA
      from ..Functions.least_square_calibration import Least_Square_Calibration
      solver = SPSA(
        #lambda_init=100,
        #lambda_factors=[0.07, 1.2, 0.7, 1.2, 70, 120],
        maxiter=max_it,
        bounds=np.repeat([density_parameter_bounds],len(initial_guess),axis=0).T,
        callback=self.scipy_callback,
      )
      if isinstance(self.obj, Least_Square_Calibration):
        f = lambda x: np.sqrt(np.mean(self.scipy_function_to_optimize(x)**2))
      else:
        f = self.scipy_function_to_optimize
      res = solver.fit(f, x0=initial_guess)
      p_opt = res["x"]

    elif optimizer == 'lmfit':
      import lmfit
      params = lmfit.Parameters()
      for i in range(len(initial_guess)):
        params.add(
          f"p{i}", value=initial_guess[i],
          min=density_parameter_bounds[0], max=density_parameter_bounds[1]
        )
      minimizer = lmfit.Minimizer(
        lambda p: self.scipy_function_to_optimize([p[k].value for k in p.keys()]),
        params,
        #jac = (lambda p: self.scipy_jac([p[k].value for k in p.keys()]))
      )
      def lm_callback(params, iter, resid, *args, **kwargs):
        chi2 = 0.5 * np.dot(resid, resid)
        print(f"[Iter {iter}] Chi2={chi2:.6e}")
        #history.append(params.valuesdict().copy())
        return 0  # return nonzero to stop early
      result = minimizer.minimize(method='leastsq', callback=lm_callback)
      print(result)

    else:
      print(f"Error: Unknown optimizer or unadapted \"{optimizer}\"")
      return None
      
    #print last iteration
    print("\nOptimization finished!")
    print(f"Best {self.obj.name} cost function value: {self.best_p[0]}")
    print("Write optimum")
    print("END!")
    
    # Review this with best value
    out = Output_Struct(self.best_p[1])
    if self.filters: 
      out.p_opt_filtered = self.filter_sequence.filter(p_opt)
    out.fx = self.best_p[0]
    out.grad_fx = self.last_grad
    out.cx = self.last_constraints
    out.grad_cx = self.last_grad_constraints
    out.func_eval = self.func_eval
    out.func_eval_history = self.func_eval_history
    out.mat_props = self.get_material_properties_from_p(self.last_p_bar)
    return out
    
  
  ######################
  ### NLOPT FUNCTION ###
  ######################
  def nlopt_function_to_optimize(self, p, grad=np.array([])):
    self.output_to_user()
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
    self.__store_best_p__(p,cf)
    # output to user: in nlopt, there is no callback, so we write the results here
    # but constraints are evaluated after, so call it here
    for c in self.constraints:
      the_constraint,compare,val = c
      constraint = the_constraint.evaluate(p_bar)
      self.last_constraints[the_constraint.name] = constraint
    return cf

  def nlopt_constraint(self, i):
    """
    Function defining the ith constraints to pass to nlopt "set_(in)equality_constraint()"
    """
    return functools.partial(self.__nlopt_generic_constraint_to_optimize__, iconstraint=i)
  
  def __nlopt_generic_constraint_to_optimize__(self, p, grad=np.array([]), iconstraint=0):
    ###FILTERING: convert p to p_bar
    if not np.allclose(p, self.last_p, rtol=1e-06):
      self.func_eval += 1
      p_bar = self.pre_evaluation_objective(p)
    p_bar = self.filter_sequence.filter(p)
    the_constraint,compare,val = self.constraints[iconstraint]
    constraint = the_constraint.evaluate(p_bar)
    c_name = self.constraints[iconstraint][0].name
    print(f"Current {c_name} (constraint): {constraint:.6e}")
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
    if not np.allclose(p, self.last_p, rtol=1e-06):
      self.func_eval += 1
      p_bar = self.pre_evaluation_objective(p)
    else:
      diff_norm = np.linalg.norm(p-self.last_p) / np.sqrt(len(p))
      print(f"Do not rerun simulation as new p close to previously simulated p, diff norm: {diff_norm:.2e}")
      p_bar = self.last_p_bar
    return p_bar
  
  def scipy_function_to_optimize(self,p):
    print("\nNew function evaluation")
    self.last_p_bar[:] = self.scipy_run_sim(p)
    self.last_p[:] = p
    cf = self.evaluate_objective(self.last_p_bar)
    self.last_cf = cf
    self.__store_best_p__(p, cf)
    self.func_eval_history.append((self.func_eval, np.sqrt(np.mean(cf**2))))
    return cf
  
  def scipy_jac(self,p):
    self.scipy_run_sim(p)
    grad = self.evaluate_total_gradient(self.obj, self.last_p, self.last_p_bar)
    self.last_grad = grad.T
    return grad.T
  
  def scipy_constraint_val(self, constraint, p):
    self.scipy_run_sim(p)
    p_bar = self.filter_sequence.filter(p)
    val = constraint[0].evaluate(p_bar)
    self.last_constraints[constraint[0].name] = val
    if constraint[1] == '<':
      return - val + constraint[2]
    elif constraint[1] == '>':
      return val - constraint[2]
  
  def scipy_constraint_jac(self, constraint, p):
    grad = self.evaluate_total_gradient(constraint[0], p)
    self.last_grad_constraints[constraint[0].name] = grad.copy()
    if constraint[1] == '<': grad *= -1.
    return grad
  
  def scipy_callback(self, x, res=None):
    self.iteration += 1
    self.output_to_user()
    #self.func_eval_history.append((self.func_eval, self.best_p[0]))
    return

  def __store_best_p__(self, p, cf):
    cf = np.sqrt(np.mean(cf**2))
    # Check if the constraints are satisfied first
    for c,cmp,val in self.constraints:
      if cmp == "<" and not self.last_constraints[c.name] - val < 0.: return
      if cmp == ">" and not self.last_constraints[c.name] - val > 0.: return
    # Check for best value
    if self.best_p[0] is None:
      self.best_p = (cf, p.copy())
      return
    if self.action == "minimize" and cf < self.best_p[0]:
      self.best_p = (cf, p.copy())
    elif self.action == "maximize" and cf > self.best_p[0]:
      self.best_p = (cf, p.copy())
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
    
    # Filter initialisation
    # During filtering, the filter will received an array of size len(input_ids) and an array len(output_ids) to populate
    # At the end, the output array is the filtered p with correspondance p to cell ids (self.p_ids)
    # Here we determine which filter write where in this output array
    # determine not filtered cells
    output_ids = set()
    for f in self.filters:
      output_ids = output_ids.union([x for x in f.output_ids if x >= 0])
    not_filtered_cells = parameterized_cells.difference(output_ids)
    # add a unit filter for non filtered cells
    if len(not_filtered_cells): self.filters.insert(0, Unit_Filter(list(not_filtered_cells)))
    self.filter_sequence = Filter_Sequence(self.filters)
    # check if there is no more filtered cell than parametrized
    assert len(self.filter_sequence.output_ids) == len(parameterized_cells), "Seems there is more filtered cell than parametrized cell. You should not have filtered cell not parametrized."

    # Create mapping from p indexing to simulation indexing
    self.p_ids = self.filter_sequence.output_ids # Correspondance between p and id in simulation
    #self.ids_p = -np.ones(self.solver.get_grid_size()+1, dtype='i8') #+1 because it can be 0 or 1 based
    #self.ids_p[self.p_ids] = np.arange(len(self.p_ids)) #now we can query self.ids_p[cell_ids] and get index in p (or -1 if not attributed)
    self.ids_p = self.filter_sequence.sim_to_p_ids
    self.p_bar = np.zeros_like(self.p_ids)

    # Assign which mat prop should work on which p indexes
    for m in self.mat_props:
      m.indexes = np.nonzero(np.isin(self.p_ids, m.cell_ids))[0]
    
    #create correspondance and problem size
    self.problem_size = self.filter_sequence.input_dim
    #self.p = np.zeros(input_dim, dtype='f8')
    self.last_p = np.zeros(self.problem_size,dtype='f8')
    self.last_grad = np.zeros(self.problem_size,dtype='f8')
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
      self.__initialize_adjoint__(self.obj)
    if self.obj.indexes is None:
      # pass all cell
      self.obj.indexes = np.arange(len(self.solver.get_region_ids("__all__")))
      
    #self.obj.set_p_to_cell_ids(self.p_ids)
    
    #initialize adjoint for constraints
    self.last_constraints = {x[0].name:0. for x in self.constraints}
    self.last_grad_constraints = {x[0].name:None for x in self.constraints}
    for constraint in self.constraints:
      constraint = constraint[0]
      #constraint.set_p_to_cell_ids(self.p_ids)
      # set adjoint
      if constraint.adjoint is None:
        self.__initialize_adjoint__(constraint)
        # set indexes
        if constraint.indexes is None: # pass all cell
          constraint.indexes = np.arange(len(self.solver.get_region_ids("__all__")))
        
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
      f"Gradient d{x[0].__get_name__()}_dp" : "cell" for x in self.constraints
    })
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
      input_ids = f.get_input_ids()
      f.set_inputs({k:v[input_ids-self.solver.cell_id_start_at] for k,v in func_var.items() if k in f.variables_needed})
    self.filters_initialized = True
    return


  def __initialize_adjoint__(self, func):
    solved_variables_needed = []
    for var in func.__get_variables_needed__():
      var_ = var.replace("_INTERPOLATOR","")
      if var_ in self.solver.solved_variables:
        solved_variables_needed.append(var_)
    if len(solved_variables_needed) == 0:
      adjoint = No_Adjoint(self.mat_props, self.p_ids-self.solver.cell_id_start_at, self.ids_p)
    elif self.adjoint == "fd":
      f = lambda p: self.evaluate_objective(self.pre_evaluation_objective(p))
      adjoint = Sensitivity_Finite_Difference(f, **self.adjoint_args)
    elif self.adjoint == "adjoint":
      adjoint = Sensitivity_Steady_Simple(
        solved_variables_needed,
        self.mat_props,
        self.solver,
        self.p_ids-self.solver.cell_id_start_at,
        self.ids_p,
        *( (self.adjoint_args,) if self.adjoint_args is not None else () )
      )
    elif self.adjoint == "adjoint-corrected":
      f = lambda p: self.evaluate_objective(self.pre_evaluation_objective(p))
      adjoint = Sensitivity_Steady_Adjoint_Corrected(
        solved_variables_needed,
        self.mat_props,
        self.solver,
        self.p_ids-self.solver.cell_id_start_at,
        f,
        *( (self.adjoint_args,) if self.adjoint_args is not None else () ),
      )
    elif self.adjoint == "ensemble":
      f = lambda p: self.evaluate_objective(self.pre_evaluation_objective(p))
      adjoint = Sensitivity_Ensemble(f, **self.adjoint_args)
    func.set_adjoint_problem(adjoint)
    
  
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
\t       Moise Rousseau (2026)
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
    
    

