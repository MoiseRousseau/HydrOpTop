import sys
import os
path = os.getcwd() + '/../../'
sys.path.append(path)
import numpy as np

class Test_Sum_Flux:
  #create PFLOTRAN simulation object
  from HydrOpTop import PFLOTRAN
  from HydrOpTop.Functions import Sum_Flux
  from common import __add_inputs__
  
  pft_problem = "pit_3d"
  pflotranin = f"../PFLOTRAN_problems/{pft_problem}/pflotran.in"
  sim = PFLOTRAN(pflotranin)
  sim.run()
  #create sum flux object
  connection_to_sum = sim.get_connections_ids_integral_flux("interface")
  obj = Sum_Flux(connection_to_sum, option="signed")
  __add_inputs__(obj, sim)
    
  def test_sum_flux_signed_vs_PFLOTRAN_integral_flux(self):
    self.obj.option = "signed"
    #get results from integral flux
    src = open(f"../PFLOTRAN_problems/{self.pft_problem}/pflotran-int.dat", 'r')
    line = src.readlines()[-1]
    flux_pft = float(line.split()[3]) / (3600*24*365) / 997.16
    src.close()
    #calculate flux using sum flux object
    flux_obj = self.obj.evaluate(None)
    #compare
    diff = abs(1-flux_obj/flux_pft)
    print(f"Signed PFLOTRAN flux: {flux_pft:.6e} m3/s")
    print(f"Signed Sum_Flux flux: {flux_obj:.6e} m3/s")
    print(f"Relative difference: {diff:.6e}")
    assert diff < 2e-5
  
  def test_sum_flux_signed_vs_PFLOTRAN_integral_flux_absolute(self):
    self.obj.option = "absolute"
    #run PFLOTRAN and get results from integral flux
    src = open(f"../PFLOTRAN_problems/{self.pft_problem}/pflotran-int.dat", 'r')
    line = src.readlines()[-1]
    flux_pft = float(line.split()[5]) / (3600*24*365) / 997.16
    src.close()
    #calculate flux using sum flux object
    flux_obj = self.obj.evaluate(None)
    #compare
    diff = abs(1-flux_obj/flux_pft)
    print(f"PFLOTRAN flux: {flux_pft:.6e} m3/s")
    print(f"Sum_Flux flux: {flux_obj:.6e} m3/s")
    print(f"Relative difference: {diff:.6e}")
    assert diff < 2e-5
  
  def FD_dP(self, ids_to_test):
    ref_val = self.obj.evaluate(None)
    pressure = self.obj.pressure
    d_obj_fd = np.zeros(len(ids_to_test),dtype='f8')
    pertub = 1e-6
    for i,cell_id in enumerate(ids_to_test):
      old_pressure = pressure[cell_id-1]
      d_pressure = old_pressure * pertub
      pressure[cell_id-1] = old_pressure + d_pressure
      d_obj_fd[i] = (self.obj.evaluate(None)-ref_val) / d_pressure
      pressure[cell_id-1] = old_pressure
    return d_obj_fd
  
  def test_derivative_dP_signed(self):
    self.obj.option = "signed"
    ids_to_test = np.unique(self.connection_to_sum.flatten())
    #get analytic derivative
    d_obj = self.obj.d_objective_dY(None)[0]
    #get finite difference derivative
    d_obj_fd = self.FD_dP(ids_to_test)
    #perform comparison
    distance = np.sqrt(np.sum((d_obj_fd-d_obj[ids_to_test-1])**2)) / np.sqrt(np.sum(d_obj_fd**2))
    print(f"Relative distance in the {len(d_obj_fd)}-dimensional space: {distance}")
    print("Cell, derivative FD, derivative analytic")
    for i in range(len(d_obj_fd)):
      print(f"{i+1}, {d_obj_fd[i]:.6e}, {d_obj[ids_to_test[i]-1]:.6e}")
    assert distance < 1e-6
    return
    
  def test_derivative_dP_absolute(self):
    self.obj.option = "absolute"
    ids_to_test = np.unique(self.connection_to_sum.flatten())
    #get analytic derivative
    d_obj = self.obj.d_objective_dY(None)[0]
    #get finite difference derivative
    d_obj_fd = self.FD_dP(ids_to_test)
    #perform comparison
    distance = np.sqrt(np.sum((d_obj_fd-d_obj[ids_to_test-1])**2)) / np.sqrt(np.sum(d_obj_fd**2))
    print(f"Relative distance in the {len(d_obj_fd)}-dimensional space: {distance}")
    print("Cell, derivative FD, derivative analytic")
    for i in range(len(d_obj_fd)):
      print(f"{i+1}, {d_obj_fd[i]:.6e}, {d_obj[ids_to_test[i]-1]:.6e}")
    assert distance < 5e-5
    return
  
  
  def FD_dK(self, ids_to_test):
    ref_val = self.obj.evaluate(None)
    permeability = self.obj.k
    d_obj_fd = np.zeros(len(ids_to_test),dtype='f8')
    pertub = 1e-6
    for i,cell_id in enumerate(ids_to_test):
      old_perm = permeability[cell_id-1]
      d_perm = old_perm * pertub
      permeability[cell_id-1] = old_perm + d_perm
      d_obj_fd[i] = (self.obj.evaluate(None)-ref_val) / d_perm
      permeability[cell_id-1] = old_perm
    return d_obj_fd
  
  def test_derivative_dK_signed(self):
    self.obj.option = "signed"
    ids_to_test = np.unique(self.connection_to_sum.flatten())
    #get analytic derivative
    d_obj = self.obj.d_objective_dX(None)[1]
    #get finite difference derivative
    d_obj_fd = self.FD_dK(ids_to_test)
    #perform comparison
    distance = np.sqrt(np.sum((d_obj_fd-d_obj[ids_to_test-1])**2)) / np.sqrt(np.sum(d_obj_fd**2))
    print(f"Relative distance in the {len(d_obj_fd)}-dimensional space: {distance}")
    print("Cell, derivative FD, derivative analytic")
    for i in range(len(d_obj_fd)):
      print(f"{i+1}, {d_obj_fd[i]:.6e}, {d_obj[ids_to_test[i]-1]:.6e}")
    assert distance < 1e-6
    return
    
  def test_derivative_dK_absolute(self):
    self.obj.option = "absolute"
    ids_to_test = np.unique(self.connection_to_sum.flatten())
    #get analytic derivative
    d_obj = self.obj.d_objective_dX(None)[1]
    #get finite difference derivative
    d_obj_fd = self.FD_dK(ids_to_test)
    #perform comparison
    distance = np.sqrt(np.sum((d_obj_fd-d_obj[ids_to_test-1])**2)) / np.sqrt(np.sum(d_obj_fd**2))
    print(f"Relative distance in the {len(d_obj_fd)}-dimensional space: {distance}")
    print("Cell, derivative FD, derivative analytic")
    for i in range(len(d_obj_fd)):
      print(f"{i+1}, {d_obj_fd[i]:.6e}, {d_obj[ids_to_test[i]-1]:.6e}")
    assert distance < 1e-6
    return
  
