import sys
import os
path = os.getcwd() + '/../../'
sys.path.append(path)

import numpy as np
from HydrOpTop import PFLOTRAN
from HydrOpTop.Functions import Head_Gradient

class Test_Head_Gradient:
  #create PFLOTRAN simulation object
  pft_problem = "9x9x1"
  pflotranin = f"../PFLOTRAN_problems/{pft_problem}/source_sink_center.in"
  sim_ss = PFLOTRAN(pflotranin)
  sim_ss.run_PFLOTRAN()
  
  pft_problem = "9x9x1"
  pflotranin = f"../PFLOTRAN_problems/{pft_problem}/uniform_flow.in"
  sim_uniform = PFLOTRAN(pflotranin)
  sim_uniform.run_PFLOTRAN()
  
  pft_problem = "pit_3d"
  pflotranin = f"../PFLOTRAN_problems/{pft_problem}/pflotran.in"
  sim_exp_grid = PFLOTRAN(pflotranin)
  perm_data = np.genfromtxt(f"../PFLOTRAN_problems/{pft_problem}/permeability_field.csv",
                             comments='#')
  cell_ids, perm_field = perm_data[:,0], perm_data[:,1]
  sim_exp_grid.create_cell_indexed_dataset(perm_field, "permeability", "permeability.h5", cell_ids)
  sim_exp_grid.run_PFLOTRAN()
  
  
  def test_value_source_sink(self):
    """
    Test the gradient versus a trivial value
    """
    #initiate objective
    center_id = self.sim_ss.get_region_ids("center")
    obj = Head_Gradient(ids_to_consider=center_id, power=1) #the center cell id where the sink is
    inputs = []
    for output in obj.__get_PFLOTRAN_output_variable_needed__():
      if output == "CONNECTION_IDS": 
        inputs.append(self.sim_ss.get_internal_connections())
        continue
      inputs.append(self.sim_ss.get_output_variable(output))
    obj.set_inputs(inputs)
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
    center_id = self.sim_uniform.get_region_ids("center")
    obj = Head_Gradient(power=1) #the center cell id where the sink is
    inputs = []
    for output in obj.__get_PFLOTRAN_output_variable_needed__():
      if output == "CONNECTION_IDS": 
        inputs.append(self.sim_uniform.get_internal_connections())
        continue
      inputs.append(self.sim_uniform.get_output_variable(output))
    obj.set_inputs(inputs)
    obj.set_p_to_cell_ids(np.arange(10,73))
    cf = obj.evaluate(None)
    print(f"Head Gradient: {cf} (should be very close to 0.01)")
    assert abs(cf-0.01) < 1e-6
  
  
  def test_derivative_dP(self):
    """
    Test analytical derivative compared to finite difference.
    Do not choose the cell near the source sink as the finite difference derivative
    does not seem to converge.
    """
    #set up Head_Gradient object
    obj = Head_Gradient(ids_to_consider=np.arange(1,41), power=1.) 
    inputs = []
    for output in obj.__get_PFLOTRAN_output_variable_needed__():
      if output == "CONNECTION_IDS": 
        inputs.append(self.sim_ss.get_internal_connections())
        continue
      inputs.append(self.sim_ss.get_output_variable(output))
    obj.set_inputs(inputs)
    obj.set_p_to_cell_ids(np.arange(1,82))
    #get analytic derivative
    obj.d_objective_dP(None)
    d_obj = obj.dobj_dP
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
    obj = Head_Gradient(ids_to_consider=pit_ids, power=1.) 
    inputs = []
    for output in obj.__get_PFLOTRAN_output_variable_needed__():
      if output == "CONNECTION_IDS": 
        inputs.append(self.sim_exp_grid.get_internal_connections())
        continue
      inputs.append(self.sim_exp_grid.get_output_variable(output))
    obj.set_inputs(inputs)
    obj.set_p_to_cell_ids(np.arange(1,self.sim_exp_grid.n_cells+1))
    #get analytic derivative
    obj.d_objective_dP(None)
    d_obj = obj.dobj_dP
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
  
  
