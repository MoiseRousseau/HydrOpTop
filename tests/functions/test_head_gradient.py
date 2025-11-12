import sys
import os
path = os.getcwd() + '/../../'
sys.path.append(path)

import numpy as np
from HydrOpTop.Solvers import PFLOTRAN
from HydrOpTop.Functions import Head_Gradient
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
  
  
  def test_value_source_sink(self):
    """
    Test the gradient versus a trivial value
    """
    #initiate objective
    center_id = self.sim_ss.get_region_ids("center")
    obj = Head_Gradient(ids_to_consider=center_id, power=1) #the center cell id where the sink is
    __add_inputs__(obj, self.sim_ss)
    obj.set_p_to_cell_ids(np.arange(1,82))
    cf = obj.evaluate(None)
    print(f"Head Gradient: {cf} (should be very close to 0.)")
    assert cf < 1e-6
  
  
  def test_value_uniform(self):
    """
    Test the gradient versus a imposed gradient
    Does not consider the boundary cell as the corrected gradient is halfed
    """
    #initiate objective
    ids = np.arange(10,73)
    obj = Head_Gradient(ids_to_consider=ids, power=1) 
    __add_inputs__(obj, self.sim_uniform)
    obj.set_p_to_cell_ids(ids)
    cf = obj.evaluate(None)
    print(f"Head Gradient: {cf} (should be very close to 0.01)")
    print(obj.get_head_gradient())
    assert abs(cf-0.01) < 1e-6
  
  def test_value_restricted(self):
    """
    Test the gradient using the restricted domain option.
    Since only one cell, gradient should be null.
    """
    #initiate objective
    center_id = self.sim_uniform.get_region_ids("center")
    obj = Head_Gradient(ids_to_consider=center_id, power=1, restrict_domain=True) 
    __add_inputs__(obj,self.sim_ss)
    obj.set_p_to_cell_ids(np.arange(1,82))
    cf = obj.evaluate(None)
    print(f"Head Gradient: {cf} (should be very close to 0.)")
    assert abs(cf) < 1e-6
     
  
  def test_derivative_dP(self):
    """
    Test analytical derivative compared to finite difference.
    Do not choose the cell near the source sink as the finite difference derivative
    does not seem to converge.
    """
    #set up Head_Gradient object
    obj = Head_Gradient(ids_to_consider=np.arange(1,41), power=1.) 
    __add_inputs__(obj, self.sim_ss)
    obj.set_p_to_cell_ids(np.arange(1,82))
    #get analytic derivative
    d_obj = obj.d_objective_dY(None)[0]
    ref_val = obj.evaluate(None)
    #get finite difference derivative
    pressure = obj.pressure
    d_obj_fd = np.zeros(len(pressure),dtype='f8')
    pertub = 1e-9
    for i in range(len(d_obj_fd)):
      old_pressure = pressure[i]
      d_pressure = old_pressure * pertub
      pressure[i] = old_pressure + d_pressure
      d_obj_fd[i] = (obj.evaluate(None)-ref_val) / d_pressure
      pressure[i] = old_pressure
    #perform comparison
    distance = np.sqrt(np.sum((d_obj_fd-d_obj)**2)) / np.sqrt(np.sum(d_obj_fd**2))
    print(f"Relative distance in the {len(d_obj)}-dimensional space: {distance}")
    print("Cell, derivative FD, derivative analytic")
    for i in range(len(pressure)):
      print(f"{i+1}, {d_obj_fd[i]:.6e}, {d_obj[i]:.6e}")
    assert distance < 1e-6
    return
  
    
  def test_derivative_dP_exp_grid(self):
    """
    Test analytical derivative compared to finite difference using
    a explicit unstructured grid.
    """
    #set up Head_Gradient object
    pit_ids = self.sim_exp_grid.get_region_ids("pit")
    obj = Head_Gradient(ids_to_consider=pit_ids, power=2.) 
    __add_inputs__(obj, self.sim_exp_grid)
    obj.set_p_to_cell_ids(np.arange(1,self.sim_exp_grid.n_cells+1))
    #get analytic derivative
    d_obj = obj.d_objective_dY(None)[0]
    ref_val = obj.evaluate(None)
    #get finite difference derivative
    pressure = obj.pressure
    d_obj_fd = np.zeros(len(pit_ids),dtype='f8')
    pertub = 1e-9
    for i,cell_id in enumerate(pit_ids):
      old_pressure = pressure[cell_id-1]
      d_pressure = old_pressure * pertub
      pressure[cell_id-1] = old_pressure + d_pressure
      d_obj_fd[i] = (obj.evaluate(None)-ref_val) / d_pressure
      pressure[cell_id-1] = old_pressure
    #perform comparison
    distance = np.sqrt(np.sum((d_obj_fd-d_obj[pit_ids-1])**2)) / np.sqrt(np.sum(d_obj_fd**2))
    print(f"Relative distance in the {len(d_obj_fd)}-dimensional space: {distance}")
    print("Cell, derivative FD, derivative analytic")
    for i in range(len(d_obj_fd)):
      print(f"{i+1}, {d_obj_fd[i]:.6e}, {d_obj[pit_ids[i]-1]:.6e}")
    assert distance < 1e-5
    return
  
  def test_derivative_dP_restrict_domain(self):
    """
    Test analytical derivative compared to finite difference using the 
    restricted domain option
    """
    #set up Head_Gradient object
    pit_ids = self.sim_exp_grid.get_region_ids("pit")
    obj = Head_Gradient(ids_to_consider=pit_ids, restrict_domain=True) 
    __add_inputs__(obj, self.sim_exp_grid)
    obj.set_p_to_cell_ids(np.arange(1,self.sim_exp_grid.n_cells+1))
    #get analytic derivative
    d_obj = obj.d_objective_dY(None)[0]
    ref_val = obj.evaluate(None)
    #get finite difference derivative
    pressure = obj.pressure
    d_obj_fd = np.zeros(len(pit_ids),dtype='f8')
    pertub = 1e-8
    for i,cell_id in enumerate(pit_ids):
      old_pressure = pressure[cell_id-1]
      d_pressure = old_pressure * pertub
      pressure[cell_id-1] = old_pressure + d_pressure
      d_obj_fd[i] = (obj.evaluate(None)-ref_val) / d_pressure
      pressure[cell_id-1] = old_pressure
    #perform comparison
    distance = np.sqrt(np.sum((d_obj_fd-d_obj[pit_ids-1])**2)) / np.sqrt(np.sum(d_obj_fd**2))
    print(f"Relative distance in the {len(d_obj_fd)}-dimensional space: {distance}")
    print("Cell, derivative FD, derivative analytic")
    for i in range(len(d_obj_fd)):
      print(f"{i+1}, {d_obj_fd[i]:.6e}, {d_obj[pit_ids[i]-1]:.6e}")
    assert distance < 1e-5
    assert abs(d_obj[0]) < 1e-6
    return
  
  
