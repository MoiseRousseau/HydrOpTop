import sys
import os
path = os.getcwd() + '/../../'
sys.path.append(path)
import numpy as np

class Test_Sum_Flux:
  #create PFLOTRAN simulation object
  from HydrOpTop import PFLOTRAN
  from HydrOpTop.Functions import Sum_Flux
  pft_problem = "pit_3d"
  pflotranin = f"../PFLOTRAN_problems/{pft_problem}/pflotran.in"
  sim = PFLOTRAN(pflotranin)
  sim.run_PFLOTRAN()
  #create sum flux object
  connection_to_sum = sim.get_connections_ids_integral_flux("interface")
  obj = Sum_Flux(connection_to_sum)#TODO:, option="signed")
  inputs = []
  for output in obj.__get_PFLOTRAN_output_variable_needed__():
    if output == "CONNECTION_IDS": 
      inputs.append(sim.get_internal_connections())
      continue
    inputs.append(sim.get_output_variable(output))
  obj.set_inputs(inputs)
    
  def test_sum_flux_signed_vs_PFLOTRAN_integral_flux(self):
    #run PFLOTRAN and get results from integral flux
    src = open(f"../PFLOTRAN_problems/{self.pft_problem}/pflotran-int.dat", 'r')
    line = src.readlines()[-1]
    flux_pft = float(line.split()[3]) / (3600*24*365) / 997.16
    src.close()
    #calculate flux using sum flux object
    flux_obj = self.obj.evaluate(None)
    #compare
    diff = abs(1-flux_obj/flux_pft)
    print(f"PFLOTRAN flux: {flux_pft:.6e} m3/s")
    print(f"Sum_Flux flux: {flux_obj:.6e} m3/s")
    print(f"Relative difference: {diff:.6e}")
    assert diff < 2e-5
  
  
  def test_derivative_dP(self):
    return
  
  
  def test_derivative_dK(self):
    #get analytic derivative
    self.obj.d_objective_d_mat_props(None)
    d_obj = self.obj.dobj_dmat_props[2]
    ref_val = self.obj.evaluate(None)
    #get finite difference derivative
    permeability = self.obj.k
    ids_to_test = np.unique(self.connection_to_sum.flatten())
    d_obj_fd = np.zeros(len(ids_to_test),dtype='f8')
    pertub = 1e-9
    for i,cell_id in enumerate(ids_to_test):
      old_perm = permeability[cell_id-1]
      d_perm = old_perm * pertub
      permeability[cell_id-1] = old_perm + d_perm
      d_obj_fd[i] = (self.obj.evaluate(None)-ref_val) / d_perm
      permeability[cell_id-1] = old_perm
    #perform comparison
    distance = np.sqrt(np.sum((d_obj_fd-d_obj[ids_to_test-1])**2)) / np.sqrt(np.sum(d_obj_fd**2))
    print(f"Relative distance in the {len(d_obj_fd)}-dimensional space: {distance}")
    print("Cell, derivative FD, derivative analytic")
    for i in range(len(d_obj_fd)):
      print(f"{i+1}, {d_obj_fd[i]:.6e}, {d_obj[ids_to_test[i]-1]:.6e}")
    assert distance < 1e-6
    return
  
