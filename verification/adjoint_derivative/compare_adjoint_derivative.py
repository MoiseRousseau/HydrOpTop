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
from HydrOpTop.Objectives import Sum_Liquid_Piezometric_Head
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
  sens = Sensitivity_Richards([mat], pft_model,np.arange(0,128))
  sens.set_adjoint_solving_algo("lu")
  S_adjoint = sens.compute_sensitivity(0., cost_deriv_pressure, [0.])
  return S_adjoint
  


def compute_sensitivity_finite_difference(pertub = 1e-6):
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
  deriv = np.zeros(len(pressure),dtype='f8')
  for i in range(len(pressure)):
    print("Compute derivative of head sum for element {}".format(i+1))
    old_perm = perm_field[i]
    perm_field[i] += old_perm * pertub
    pft_model.create_cell_indexed_dataset(perm_field, "permeability", 
                                          "Permeability.h5", cell_ids)
    pft_model.run_PFLOTRAN()
    pft_model.get_output_variable("LIQUID_PRESSURE",out=pressure)
    cur_obj = objective.evaluate(0.)
    deriv[i] = (cur_obj-ref_obj) / (old_perm * pertub)
    perm_field[i] = old_perm
  return deriv


def make_verification():
  #make simulation
  print("\nMake sensitivity analysis using adjoint equation")
  S_adjoint = compute_sensitivity_adjoint()
  print("Make sensitivity analysis using finite difference")
  S_FD = compute_sensitivity_finite_difference()
  print("\n")
  
  #print stats to screen
  rel_diff = 1 - np.abs(S_adjoint/S_FD)
  print(f"Max relative diff (must be close to 0): {np.max(rel_diff)}")
  print("Output sensitivities in diff.txt")
  out = open("diff.txt",'w')
  out.write("Cell_Id S_adjoint S_finite_diff Relative_difference\n")
  for i in range(len(S_adjoint)):
    out.write(f"{i+1} {S_adjoint[i]} {S_FD[i]} {rel_diff[i]}\n")
  out.close()
  
  #1 mean error, 0 success
  if np.max(rel_diff) > 0.01: return 1
  else: return 0
  
  
  

if __name__ == "__main__":
  err = make_verification()
