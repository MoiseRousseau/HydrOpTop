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
    self.constrain_inputs_arrays = None
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
  
  def get_problem_size(self): return self.problem_size
    
  
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
      for var in self.obj.__get_PFLOTRAN_output_variable_needed__():
        if var == "CONNECTION_IDS":
          self.Yi.append(self.solver.get_internal_connections())
          continue
        self.Yi.append(self.solver.get_output_variable(var))
      self.obj.set_inputs(self.Yi)
    else: 
      for i,var in enumerate(self.obj.__get_PFLOTRAN_output_variable_needed__()):
        if var == "CONNECTION_IDS":
          self.solver.get_internal_connections(self.Yi[i])
          continue
        else:
          self.solver.get_output_variable(var, self.Yi[i], -1) #last timestep
      
    ### UPDATE constraints ###
    if self.constrain_inputs_arrays is None: #need to initialize
      self.constrain_inputs_arrays = []
      for constrain in self.constraints:
        temp = []
        for var in constrain.__get_PFLOTRAN_output_variable_needed__():
          if var == "CONNECTION_IDS":
            temp.append(self.solver.get_internal_connections())
            continue
          temp.append(self.solver.get_output_variable(var))
        self.constrain_inputs_arrays.append(temp)
        constrain.set_inputs(self.constrain_inputs_arrays[-1])
    else: 
      for i,constrain in enumerate(self.constraints):
        for j,var in enumerate(constrain.__get_PFLOTRAN_output_variable_needed__()):
          if var == "CONNECTION_IDS":
            self.solver.get_internal_connections(self.constrain_inputs_arrays[i][j])
            continue
          self.solver.get_output_variable(var, self.constrain_inputs_arrays[i][j], -1)
    return
    
  
  
  ### READY MADE WRAPPER FOR POPULAR LIBRARY ###
  def nlopt_function_to_optimize(self, p, grad):
    """
    Cost function to pass to NLopt method "set_min/max_objective()"
    """
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
    cf = self.obj.nlopt_optimize(p_bar,grad)
    if self.first_cf is None: self.first_cf = cf
    if self.filter and grad.size > 0:
      grad[:] = self.filter.get_filter_derivative(p).transpose().dot(grad)
    ### OUTPUT ###
    #TODO: constraint not outputed
    if self.filter is None: 
      self.IO.output(self.func_eval, 
                     cf, [0], p, grad, [0], val_at=self.p_ids-1)
      #self.IO.output(self.func_eval, cf, constraints_val, p, grad_cf, grad_constraints, p_bar)
    else:
      self.IO.output(self.func_eval, cf, [0], p, grad, [0], p_bar, val_at=self.p_ids-1)
    #normalize cf to 1
    cf /= self.first_cf
    grad /= self.first_cf
    self.last_p[:] = p
    #print to user
    print(f"Current {self.obj.name}: {cf*self.first_cf:.6e}")
    print(f"Min gradient: {np.min(grad*self.first_cf):.6e} at cell id {np.argmin(grad)+1}")
    print(f"Max gradient: {np.max(grad*self.first_cf):.6e} at cell id {np.argmax(grad)+1}")
    return cf
    
  
  def nlopt_constrain(self, i):
    """
    Function defining the ith constraints to pass to nlopt "set_(in)equality_constrain()"
    """
    return functools.partial(self.__nlopt_generic_constrain_to_optimize__, iconstrain=i)
  
  def __nlopt_generic_constrain_to_optimize__(self, p, grad, iconstrain=0):
    ###FILTERING: convert p to p_bar
    if self.filter is None:
      p_bar = p
    else: 
      for i,var in enumerate(self.filter.__get_PFLOTRAN_output_variable_needed__()):
        self.solver.get_output_variable(var, self.filter_i[i], -1) #last timestep
      p_bar = self.filter.get_filtered_density(p)
    constrain = self.constraints[iconstrain].nlopt_optimize(p_bar,grad)
    if self.filter:
      grad[:] = self.filter.get_filter_derivative(p_bar).transpose().dot(grad)
    tol = self.constraints[iconstrain].__get_constrain_tol__()
    print(f"Current {self.constraints[iconstrain].name} constrain: {constrain+tol:.6e}")
    return constrain
    
   
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
    for constrain in self.constraints:
      if constrain.__require_adjoint__() and (constrain.adjoint is None):
        which = self.obj.__require_adjoint__()
        if which == "RICHARDS":
          adjoint = Sensitivity_Richards(self.mat_props, self.solver, self.p_ids)
          constrain.set_adjoint_problem(adjoint)
      constrain.set_p_to_cell_ids(self.p_ids)
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
    

