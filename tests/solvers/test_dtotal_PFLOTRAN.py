import sys
import os
path = os.getcwd() + '/../../'
sys.path.append(path)

import numpy as np
from HydrOpTop.Solver import PFLOTRAN
import common

class Test_PFLOTRAN:

  def test_initialization(self):
    """
    Test if Solver can read PFLOTRAN input
    """
    sim = PFLOTRAN("../PFLOTRAN_problems/9x9x1/uniform_flow.in")
    sim = PFLOTRAN("../PFLOTRAN_problems/pit_3d/pflotran.in")
    
  def test_adjoint(self):
    """
    Test adjoint derivative compared to finite difference
    """
    from HydrOpTop.Functions import Mean_Liquid_Piezometric_Head
    from HydrOpTop.Materials import Identity
    from HydrOpTop.Adjoints import Sensitivity_Richards
    sim = PFLOTRAN("../PFLOTRAN_problems/9x9x1_opt/uniform_flow_opt.in")
    cf = Mean_Liquid_Piezometric_Head()
    matprop = Identity("all", "PERMEABILITY")
    S_adjoint = common.compute_sensitivity_adjoint(sim, cf, "PERMEABILITY", matprop, Sensitivity_Richards)
    cell_ids_to_test = np.arange(10,20)
    S_fd = common.compute_sensitivity_finite_difference(sim, cf, "PERMEABILITY", matprop, cell_ids_to_test=cell_ids_to_test, pertub=1e-3)
    print(S_adjoint, S_fd)
    for i,cell_id in enumerate(cell_ids_to_test-1):
      assert np.abs(S_adjoint[cell_id]-S_fd[i]) < 1e-6
     
    
   
 

