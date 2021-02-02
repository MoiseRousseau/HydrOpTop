import sys
import os
path = os.getcwd() + '/../../'
sys.path.append(path)

from HydrOpTop.Functions import Sum_Flux
from HydrOpTop import PFLOTRAN
from HydrOpTop.PFLOTRAN import default_water_density

import numpy as np

if __name__ == "__main__":
  pft_problem = "pit_3d"
  #create PFLOTRAN simulation object
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
  
  #get integral flux result
  src = open(f"../PFLOTRAN_problems/{pft_problem}/pflotran-int.dat", 'r')
  line = src.readlines()[-1]
  flux_pft = float(line.split()[3]) / (3600*24*365.25) / default_water_density
  print(f"PFLOTRAN flux: {flux_pft:.6e} m3/s")
  
  #compare with the objective sum flux
  connection_to_sum = sim.get_connections_ids_integral_flux("interface")
  #connection_to_sum = [[2,1],[5,1]]
  obj = Sum_Flux(connection_to_sum)
  inputs = []
  for output in obj.__get_PFLOTRAN_output_variable_needed__():
    if output == "CONNECTION_IDS": 
      inputs.append(sim.get_internal_connections())
      continue
    inputs.append(sim.get_output_variable(output))
  obj.set_inputs(inputs)
  flux_obj = obj.evaluate(None)
  print(f"Sum_Flux flux: {flux_obj:.6e} m3/s")
  
  diff = abs(1-flux_obj/flux_pft)
  print(f"Relative difference: {diff:.6e}")
  
  if diff < 1e-3: exit(0)
  else: exit(1)
