import sys
import os
path = os.getcwd() + '/../../'
sys.path.append(path)

from HydrOpTop.Functions import Sum_Flux
from HydrOpTop import PFLOTRAN
from HydrOpTop.PFLOTRAN import default_water_density

import numpy as np

if __name__ == "__main__":
  
  #create PFLOTRAN simulation object
  print("Create and run PFLOTRAN simulation \"a_problem\" to create function inputs")
  pflotranin = "../PFLOTRAN_problems/pit_general/pflotran.in"
  sim = PFLOTRAN(pflotranin)
  #create a random material parameter
  p = np.random.random(sim.get_grid_size())
  perm_data = np.genfromtxt("../PFLOTRAN_problems/pit_general/permeability_field.csv",
                            comments='#')
  cell_ids, perm_field = perm_data[:,0], perm_data[:,1]
  sim.create_cell_indexed_dataset(perm_field, "permeability", 
                                        "permeability.h5", cell_ids)
  sim.run_PFLOTRAN()
  
  #get integral flux result
  src = open("../PFLOTRAN_problems/pit_general/pflotran-int.dat", 'r')
  line = src.readlines()[-1]
  flux_pft = float(line.split()[-1]) / (3600*24*365.25) / default_water_density
  print(f"PFLOTRAN flux: {flux_pft:.6e} m3/s")
  
  #compare with the objective sum flux
  connection_to_sum = sim.get_connections_ids_integral_flux("barrier")
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
