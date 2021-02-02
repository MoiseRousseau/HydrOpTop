import sys
import os
path = os.getcwd() + '/../../'
sys.path.append(path)

from HydrOpTop.Functions import Sum_Flux, p_Weighted_Sum_Flux
from HydrOpTop import PFLOTRAN

import numpy as np

if __name__ == "__main__":
  
  #create PFLOTRAN simulation object
  pft_problem = "pit_3d"
  print(f"Create and run PFLOTRAN simulation \"{pft_problem}\" to create function inputs")
  pflotranin = f"../PFLOTRAN_problems/{pft_problem}/pflotran.in"
  sim = PFLOTRAN(pflotranin)
  #create a random material parameter
  perm_data = np.genfromtxt(f"../PFLOTRAN_problems/{pft_problem}/permeability_field.csv",
                            comments='#')
  cell_ids, perm_field = perm_data[:,0], perm_data[:,1]
  sim.create_cell_indexed_dataset(perm_field, "permeability", 
                                        "permeability.h5", cell_ids)
  sim.run_PFLOTRAN()
  
  #define objective
  #pit_ids = sim.get_region_ids("pit") cannot compare more than one element because dupplicated connection not considered in the current sum_flux function implementation
  pit_ids = np.array([98])
  weighted = p_Weighted_Sum_Flux(pit_ids)
  weighted.set_p_to_cell_ids(pit_ids)
  
  #prepare inputs to objective
  inputs = []
  for output in weighted.__get_PFLOTRAN_output_variable_needed__():
    if output == "CONNECTION_IDS": 
      inputs.append(sim.get_internal_connections())
      continue
    inputs.append(sim.get_output_variable(output))
    
  weighted.set_inputs(inputs)
  connection_to_sum = weighted.get_connection_ids_to_sum()
  
  #compare with the objective sum flux
  sum_flux = Sum_Flux(connection_to_sum, square=True)
  sum_flux.set_inputs(inputs)
  
  #evaluate
  flux_weighted = weighted.evaluate(np.ones(sim.n_cells,dtype='f8'))
  print(f"p_Weighted_Sum_Flux with p=1: {flux_weighted:.6e} m3/s")
  flux_sum_flux = sum_flux.evaluate(None)
  print(f"Sum_Flux (control): {flux_sum_flux:.6e} m3/s")
  
  diff = abs(1-flux_weighted/flux_sum_flux)
  print(f"Relative difference: {diff:.6e}")
  
  if diff < 1e-6: exit(0)
  else: exit(1)
