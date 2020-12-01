#
# Verification of the sensitivity computed via the adjoint equation
# versus finite difference for the Sum_Liquid_Piezometric_Head objective
#

import sys
import os
path = os.getcwd() + '/../../src/'
sys.path.append(path)

import h5py
import numpy as np


import PFLOTRAN as PFT
from Objective import Sum_Liquid_Piezometric_Head
from Sensitivity_Adjoint import Sensitivity_Richards


def compute_sensitivity_adjoint():
  #initialize model field
  print("Initialize PFLOTRAN model with the given permeability field")
  pft_model = PFT.PFLOTRAN("pflotran.in")
  perm_data = np.genfromtxt("permeability_field.csv", comments='#')
  cell_ids, perm_field = perm_data[:,0], perm_data[:,1]
  pft_model.create_cell_indexed_dataset(perm_field, "permeability", 
                                        "Permeability.h5", cell_ids)
  #run model
  print("Run model")
  pft_model.run_PFLOTRAN()
  
  #initiate objective
  print("Compute objective function")
  head = pft_model.get_output_variable("LIQUID_PRESSURE")
  z = pft_model.get_output_variable("Z_COORDINATE")
  head /= PFT.default_gravity * PFT.default_water_density
  head -= z
  objective = Sum_Liquid_Piezometric_Head(head)
  print(f"Objective: {objective.evaluate()}")
  
  #compute sensitivity
  print("Compute sensitivity")
  cost_deriv_pressure = objective.d_objective_dh()
  cost_deriv_pressure /= PFT.default_gravity * PFT.default_water_density
  #mat_prop_deriv_mat_parameter is unit
  #cost_deriv_mat_prop is null
  res_deriv_mat_prop= pft_model.get_sensitivity("PERMEABILITY") #Pa.s / m
  res_deriv_pressure = pft_model.get_sensitivity("PRESSURE") #[-]
  sens = Sensitivity_Richards(cost_deriv_pressure,
                              [1.],
                              None,
                              [res_deriv_mat_prop],
                              res_deriv_pressure)
  sens.set_adjoint_solving_algo("ilu")
  S_adjoint = sens.compute_sensitivity()
  return S_adjoint
  


def compute_sensitivity_finite_difference(pertub = 1e-6):
  #initiate data for calculating finite difference
  pft_model = PFT.PFLOTRAN("pflotran.in")
  perm_data = np.genfromtxt("permeability_field.csv", comments='#')
  cell_ids, perm_field = perm_data[:,0], perm_data[:,1]
  pft_model.create_cell_indexed_dataset(perm_field, "permeability", 
                                        "Permeability.h5", cell_ids)
  head = pft_model.initiate_output_cell_variable()
  objective = Sum_Liquid_Piezometric_Head(head)
  
  #run model for current objective
  pft_model.run_PFLOTRAN()
  pft_model.get_output_variable("LIQUID_PRESSURE",out=head)
  z = pft_model.get_output_variable("Z_COORDINATE")
  head /= PFT.default_gravity * PFT.default_water_density
  head -= z
  ref_obj = objective.evaluate()
  print(f"Current objective: {ref_obj}")
  
  #run finite difference
  deriv = np.zeros(len(head),dtype='f8')
  for i in range(len(head)):
    print("Compute derivative of head sum for element {}".format(i+1))
    old_perm = perm_field[i]
    perm_field[i] += old_perm * pertub
    pft_model.create_cell_indexed_dataset(perm_field, "permeability", 
                                          "Permeability.h5", cell_ids)
    pft_model.run_PFLOTRAN()
    pft_model.get_output_variable("LIQUID_PRESSURE",out=head)
    head /= PFT.default_gravity * PFT.default_water_density
    head -= z
    cur_obj = objective.evaluate()
    deriv[i] = (cur_obj-ref_obj) / (old_perm * pertub)
    perm_field[i] = old_perm
  return deriv



def print_stat(S_adjoint, S_FD):
  rel_diff = 1 - np.abs(S_adjoint/S_FD)
  print(f"Max relative diff (must be close to 0): {np.max(rel_diff)}")
  print("Output sensitivities in diff.txt")
  out = open("diff.txt",'w')
  out.write("Cell_Id S_adjoint S_finite_diff Relative_difference\n")
  for i in range(len(S_adjoint)):
    out.write(f"{i+1} {S_adjoint[i]} {S_FD[i]} {rel_diff[i]}\n")
  out.close()
  return
  


if __name__ == "__main__":
  print("Make sensitivity analysis using finite difference")
  S_FD = compute_sensitivity_finite_difference()
  print("\nMake sensitivity analysis using adjoint equation")
  S_adjoint = compute_sensitivity_adjoint()
  print("\n")
  print_stat(S_adjoint, S_FD)
  exit(0)
