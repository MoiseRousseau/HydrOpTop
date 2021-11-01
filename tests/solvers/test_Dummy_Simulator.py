import sys
import os
path = os.getcwd() + '/../../'
sys.path.append(path)

import numpy as np
from HydrOpTop.Solver import Dummy_Simulator

class Test_Dummy_Simulator:
  
  rgn = np.random.default_rng(232)
  n = 100
  A = rgn.random(n)
  b = rgn.random(n)

  def test_value_returned(self):
    solver = Dummy_Simulator()
    solver.create_cell_indexed_dataset(self.A,"A")
    solver.create_cell_indexed_dataset(self.b,"b")
    solver.run()
    assert np.allclose(solver.value, self.b/self.A)
  
  def test_sensitivity_A(self):
    #linear system R = Ax-b, dR/dA
    solver = Dummy_Simulator()
    solver.create_cell_indexed_dataset(self.A,"A")
    solver.create_cell_indexed_dataset(self.b,"b")
    solver.run()
    A_0_sens = solver.get_sensitivity("A").data[0]
    R_ref = self.A*solver.value - self.b
    self.A[0] += 1e-6
    assert np.isclose((self.A*solver.value - self.b - R_ref)[0]/1e-6, A_0_sens)
    
  def test_sensitivity_b(self):
    #linear system R = Ax-b, dR/dA
    solver = Dummy_Simulator()
    solver.create_cell_indexed_dataset(self.A,"A")
    solver.create_cell_indexed_dataset(self.b,"b")
    solver.run()
    b_0_sens = solver.get_sensitivity("b").data[0]
    R_ref = self.A*solver.value - self.b
    self.b[0] += 1e-6
    assert np.isclose((self.A*solver.value - self.b - R_ref)[0]/1e-6, b_0_sens)
  
  def test_sensitivity_x(self):
    #linear system R = Ax-b, dR/dA
    solver = Dummy_Simulator()
    solver.create_cell_indexed_dataset(self.A,"A")
    solver.create_cell_indexed_dataset(self.b,"b")
    solver.run()
    x_0_sens = solver.get_sensitivity("x").data[0]
    R_ref = self.A*solver.value - self.b
    solver.value[0] += 1e-6
    assert np.isclose((self.A*solver.value - self.b - R_ref)[0]/1e-6, x_0_sens)
   
 
    
    
  
  
