#
# Verification of the sensitivity computed via the adjoint equation
# versus finite difference for the Sum_Liquid_Piezometric_Head objective
#

import sys
import os
path = os.getcwd() + '/../../'
sys.path.append(path)

import h5py
import numpy as np


from HydrOpTop import PFLOTRAN
from HydrOpTop.Functions import Sum_Liquid_Piezometric_Head
from HydrOpTop.Adjoints import Sensitivity_Richards
from HydrOpTop.Materials import Permeability


def compute_sensitivity_adjoint():
  #initialize model field
  print("Initialize PFLOTRAN model with the given permeability field")
  pft_model = PFLOTRAN("pflotran.in")
  #pft_model.set_parallel_calling_command(4,"mpiexec.mpich")
  perm_data = np.genfromtxt("permeability_field.csv", comments='#')
  cell_ids, perm_field = perm_data[:,0], perm_data[:,1]
  pft_model.create_cell_indexed_dataset(perm_field, "permeability", 
                                        "Permeability.h5", cell_ids)
  #run model
  print("Run model")
  pft_model.run_PFLOTRAN()
  
  #initiate objective
  print("Compute objective function")
  pressure = pft_model.get_output_variable("LIQUID_PRESSURE")
  z = pft_model.get_output_variable("Z_COORDINATE")
  objective = Sum_Liquid_Piezometric_Head()
  objective.set_inputs([pressure,z])
  print(f"Objective: {objective.evaluate(0.)}")
  
  #compute sensitivity
  print("Compute sensitivity")
  objective.d_objective_dP(0.)
  cost_deriv_pressure = objective.dobj_dP
  #mat_prop_deriv_mat_parameter is unit
  #cost_deriv_mat_prop is null
  mat = Permeability([1.,2.], "everywhere", power=1)
  sens = Sensitivity_Richards([mat], pft_model,None)
  sens.set_adjoint_solving_algo("lu")
  S_adjoint = sens.compute_sensitivity(0., cost_deriv_pressure, [0.])
  return S_adjoint
  


def compute_sensitivity_finite_difference(cell_ids_to_test=None, pertub = 1e-6):
  #initiate data for calculating finite difference
  pft_model = PFLOTRAN("pflotran.in")
  perm_data = np.genfromtxt("permeability_field.csv", comments='#')
  cell_ids, perm_field = perm_data[:,0], perm_data[:,1]
  pft_model.create_cell_indexed_dataset(perm_field, "permeability", 
                                        "Permeability.h5", cell_ids)
  
  #run model for current objective
  pft_model.run_PFLOTRAN()
  pressure = pft_model.get_output_variable("LIQUID_PRESSURE")
  z = pft_model.get_output_variable("Z_COORDINATE")
  objective = Sum_Liquid_Piezometric_Head()
  objective.set_inputs([pressure,z])
  ref_obj = objective.evaluate(0.)
  print(f"Current objective: {ref_obj}")
  
  #run finite difference
  if cell_ids_to_test is None: cell_ids_to_test = np.arange(1,len(pressure)+1)
  deriv = np.zeros(len(cell_ids_to_test),dtype='f8')
  for i,cell in enumerate(cell_ids_to_test):
    print(f"Compute derivative of head sum for element {cell}")
    old_perm = perm_field[cell-1]
    perm_field[cell-1] += old_perm * pertub
    pft_model.create_cell_indexed_dataset(perm_field, "permeability", 
                                          "Permeability.h5", cell_ids)
    pft_model.run_PFLOTRAN()
    pft_model.get_output_variable("LIQUID_PRESSURE",out=pressure)
    cur_obj = objective.evaluate(0.)
    deriv[i] = (cur_obj-ref_obj) / (old_perm * pertub)
    perm_field[cell-1] = old_perm
  return deriv


def make_verification(cell_ids=None):
  #length is the number of cell to compare
  #make simulation
  print("\nMake sensitivity analysis using adjoint equation")
  S_adjoint = compute_sensitivity_adjoint()
  print("Make sensitivity analysis using finite difference")
  S_FD = compute_sensitivity_finite_difference(cell_ids)
  print("")
  
  #print stats to screen
  if cell_ids is not None:
    rel_diff = 1 - S_adjoint[[x-1 for x in cell_ids]]/S_FD
  else:
    rel_diff = 1 - S_adjoint/S_FD
    
  #out = open("diff.txt",'w')
  print("Cell_Id S_adjoint S_finite_diff Relative_difference")
  for i in range(len(S_FD)):
    if cell_ids is None: cell = i+1
    else: cell = cell_ids[i]
    print(f"{cell} {S_adjoint[cell-1]:.6e} {S_FD[i]:.6e} {rel_diff[i]:.6e}")
  #out.close()
  print(f"\nMax relative diff (must be close to 0): {np.max(abs(rel_diff))}")
  
  #1 mean error, 0 success
  if np.max(abs(rel_diff)) > 0.01: return 1
  else: return 0
  
  
  

if __name__ == "__main__":
  err = make_verification([15,22,34,69,78,104,125])
  if err: exit(1)
  else: exit(0)
