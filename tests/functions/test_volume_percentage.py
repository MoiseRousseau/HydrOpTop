import sys
import os
path = os.getcwd() + '/../../'
sys.path.append(path)

import numpy as np
from HydrOpTop.Functions import Volume_Percentage
from HydrOpTop.Solvers import PFLOTRAN


def create_test_model():
  
  pft_problem = "PFLOTRAN_pit_3d"
  pflotranin = f"tests/test_examples/{pft_problem}/pflotran.in"
  sim_exp_grid = PFLOTRAN(pflotranin)
  perm_data = np.genfromtxt(f"tests/test_examples/{pft_problem}/permeability_field.csv",
                             comments='#')
  cell_ids, perm_field = perm_data[:,0], perm_data[:,1]
  sim_exp_grid.create_cell_indexed_dataset(perm_field, "permeability", "permeability.h5", cell_ids)
  sim_exp_grid.run()
  
  pit_ids = sim_exp_grid.get_region_ids("pit")
  obj = Volume_Percentage(pit_ids)
  obj.set_inputs([sim_exp_grid.get_output_variable("VOLUME")])
  return pit_ids, obj


class Test_Common_Function:
  
  def test_value(self):
    pit_ids, obj = create_test_model()
    print(obj.V)
    p = np.zeros(len(pit_ids), dtype='f8')+0.14
    assert abs(obj.evaluate(p)-0.14) < 1e-9
  
  def test_derivative_dp_partial(self):
    pit_ids, obj = create_test_model()
    p = np.random.random(len(pit_ids))
    obj.d_objective_dp_partial(p)
    
    
  
  
