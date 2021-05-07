import sys
import os
path = os.getcwd() + '/../../'
sys.path.append(path)

import numpy as np
from HydrOpTop.Functions import Volume_Percentage
from HydrOpTop import PFLOTRAN

class Test_Common_Function:
  
  pft_problem = "pit_3d"
  pflotranin = f"../PFLOTRAN_problems/{pft_problem}/pflotran.in"
  sim_exp_grid = PFLOTRAN(pflotranin)
  perm_data = np.genfromtxt(f"../PFLOTRAN_problems/{pft_problem}/permeability_field.csv",
                             comments='#')
  cell_ids, perm_field = perm_data[:,0], perm_data[:,1]
  sim_exp_grid.create_cell_indexed_dataset(perm_field, "permeability", "permeability.h5", cell_ids)
  sim_exp_grid.run_PFLOTRAN()
  
  pit_ids = sim_exp_grid.get_region_ids("pit")
  obj = Volume_Percentage(pit_ids, 0.15)
  obj.set_inputs([sim_exp_grid.get_output_variable("VOLUME")])
  
  def test_value(self):
    print(self.obj.V)
    p = np.zeros(len(self.pit_ids), dtype='f8')+0.14
    assert abs(self.obj.evaluate(p)+0.01) < 1e-9
  
  def test_derivative_dp_partial(self):
    p = np.random.random(len(self.pit_ids))
    self.obj.d_objective_dp_partial(p)
    
    
  
  
