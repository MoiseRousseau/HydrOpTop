import numpy as np
from HydrOpTop.Solvers import PFLOTRAN
from HydrOpTop.Functions import p_Weighted_Head_Gradient
from common import __add_inputs__

class Test_Head_Gradient:
  #run PFLOTRAN simulation to set up the tests
  pft_problem = "PFLOTRAN_9x9x1"
  pflotranin = f"tests/test_examples/{pft_problem}/source_sink_center.in"
  sim_ss = PFLOTRAN(pflotranin)
  sim_ss.run()
  
  pft_problem = "PFLOTRAN_9x9x1"
  pflotranin = f"tests/test_examples/{pft_problem}/uniform_flow.in"
  sim_uniform = PFLOTRAN(pflotranin)
  sim_uniform.run()
  
  pft_problem = "PFLOTRAN_pit_3d"
  pflotranin = f"tests/test_examples/{pft_problem}/pflotran.in"
  sim_exp_grid = PFLOTRAN(pflotranin)
  perm_data = np.genfromtxt(f"tests/test_examples/{pft_problem}/permeability_field.csv",
                             comments='#')
  cell_ids, perm_field = perm_data[:,0], perm_data[:,1]
  sim_exp_grid.create_cell_indexed_dataset(perm_field, "permeability", "permeability.h5", cell_ids)
  sim_exp_grid.run()
  
  
  def test_value_uniform(self):
    """
    Test the gradient versus a imposed gradient
    Does not consider the boundary cell as the corrected gradient is halfed
    """
    #initiate objective
    center_id = self.sim_uniform.get_region_ids("center")
    obj = p_Weighted_Head_Gradient(power=1) 
    __add_inputs__(obj,self.sim_uniform)
    obj.set_p_to_cell_ids(np.arange(10,73))
    p = np.zeros(63, dtype='f8')+0.3
    cf = obj.evaluate(p)
    print(f"Head Gradient: {cf} (should be very close to 0.01)")
    assert abs(cf-0.01) < 1e-6
  
  
  
  def common_test_derivative_dP(self, obj):
    __add_inputs__(obj,self.sim_ss)
    obj.set_p_to_cell_ids(np.arange(1,82))
    #get analytic derivative
    p = np.random.random(82)
    d_obj = obj.d_objective_dY(p)[0]
    ref_val = obj.evaluate(p)
    #get finite difference derivative
    pressure = obj.pressure
    d_obj_fd = np.zeros(len(pressure),dtype='f8')
    pertub = 1e-9
    for i in range(len(d_obj_fd)):
      old_pressure = pressure[i]
      d_pressure = old_pressure * pertub
      pressure[i] = old_pressure + d_pressure
      d_obj_fd[i] = (obj.evaluate(p)-ref_val) / d_pressure
      pressure[i] = old_pressure
    #perform comparison
    distance = np.sqrt(np.sum((d_obj_fd-d_obj)**2)) / np.sqrt(np.sum(d_obj_fd**2))
    print(f"Relative distance in the {len(d_obj)}-dimensional space: {distance}")
    print("Cell, derivative FD, derivative analytic")
    for i in range(len(pressure)):
      print(f"{i+1}, {d_obj_fd[i]:.6e}, {d_obj[i]:.6e}")
    assert distance < 1e-6
    return
  
  def test_derivative_dP(self):
    """
    Test analytical derivative compared to finite difference.
    Do not choose the cell near the source sink as the finite difference derivative
    does not seem to converge.
    """
    #set up Head_Gradient object
    obj = p_Weighted_Head_Gradient(ids_to_consider=np.arange(1,41), power=1.) 
    self.common_test_derivative_dP(obj)
    return
    
  def test_derivative_dP_restricted(self):
    """
    Same as previous test but with restricted domain
    """
    obj = p_Weighted_Head_Gradient(ids_to_consider=np.arange(1,41), power=2., invert_weighting=True, restrict_domain=True) 
    self.common_test_derivative_dP(obj)
    return
  
  def test_derivative_dP_invert_weighting(self):
    """
    Same as 2nd previous test but with invert weighting and power
    """
    obj = p_Weighted_Head_Gradient(ids_to_consider=np.arange(1,41), power=2., invert_weighting=True) 
    self.common_test_derivative_dP(obj)
    return
  
    
  
  
  def common_test_derivative_dp_partial(self, obj):
    param_cells = obj.ids_to_consider+1
    __add_inputs__(obj,self.sim_exp_grid)
    obj.set_p_to_cell_ids(param_cells)
    p = np.random.random(len(param_cells))
    #get analytic derivative
    obj.d_objective_dp_partial(p)
    d_obj = obj.dobj_dp_partial
    ref_val = obj.evaluate(p)
    #FD derivative
    d_obj_fd = np.zeros(len(p),dtype='f8')
    pertub = 1e-6
    for i in range(len(p)):
      old_p = p[i]
      d_p = old_p * pertub
      p[i] = old_p + d_p
      d_obj_fd[i] = (obj.evaluate(p)-ref_val) / d_p
      p[i] = old_p
    #perform comparison
    distance = np.sqrt(np.sum((d_obj_fd-d_obj)**2)) / np.sqrt(np.sum(d_obj_fd**2))
    print(f"Relative distance in the {len(d_obj)}-dimensional space: {distance}")
    print("Cell, derivative FD, derivative analytic")
    for i in range(len(p)):
      print(f"{obj.p_ids[i]}, {d_obj_fd[i]:.6e}, {d_obj[i]:.6e}")
    assert distance < 1e-4
    return
    
  def test_derivative_dp_partial(self):
    """
    Test analytical derivative w.r.t. density parameter p
    compared to finite difference using a explicit unstructured grid.
    Use a random parameter p so test may fail randomly...
    """
    pit_ids = self.sim_exp_grid.get_region_ids("pit")
    obj = p_Weighted_Head_Gradient(ids_to_consider=pit_ids, power=2., restrict_domain=True) 
    self.common_test_derivative_dp_partial(obj)
    return
    
  def test_derivative_dp_partial_invert_weighting(self):
    """
    Same as previous test but with invert weighting and power
    """
    pit_ids = self.sim_exp_grid.get_region_ids("pit")
    obj = p_Weighted_Head_Gradient(ids_to_consider=pit_ids, power=1., invert_weighting=True) 
    self.common_test_derivative_dp_partial(obj)
    return
    
  
