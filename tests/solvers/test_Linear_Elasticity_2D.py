import numpy as np
from HydrOpTop.Solvers import Linear_Elasticity_2D
import common_test_solver

class Test_Total_Derivative:
  
  rgn = np.random.default_rng(1090)

  def test_total_derivative(self):
    """
    Test adjoint derivative compared to finite difference
    Comparison very sensitive to pertubation
    """
    from HydrOpTop.Functions import Mechanical_Compliance
    from HydrOpTop.Materials import Identity
    from HydrOpTop.Adjoints import Sensitivity_Steady_Simple
    sim = Linear_Elasticity_2D("test_examples/Lin_Elas_2D_cantilever/cantilever")
    n = sim.get_grid_size()
    cf = Mechanical_Compliance()
    matprop = Identity("all", "YOUNG_MODULUS")
    S_adjoint = common_test_solver.compute_sensitivity_adjoint(sim, cf, "YOUNG_MODULUS", matprop, Sensitivity_Steady_Simple, p=np.ones(n,dtype='f8'))
    cell_ids_to_test = np.arange(20,30)
    S_fd = common_test_solver.compute_sensitivity_finite_difference(sim, cf, "YOUNG_MODULUS", matprop, cell_ids_to_test=cell_ids_to_test, pertub=1e-2, p=np.ones(n,dtype='f8'))
    print("Sensitivity adjoint:")
    print(S_adjoint[cell_ids_to_test-1]), 
    print("Sensitivity finite diff:")
    print(S_fd)
    for i,cell_id in enumerate(cell_ids_to_test-1):
      assert np.abs(S_adjoint[cell_id]/S_fd[i]-1) < 5e-2
   
 
    
    
  
  
