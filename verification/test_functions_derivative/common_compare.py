
import sys
import os
path = os.getcwd() + '/../../'
sys.path.append(path)

import numpy as np
import inspect


import HydrOpTop.Functions as Functions
#import HydrOpTop.Materials as Materials
from HydrOpTop import PFLOTRAN


def common_compare(debug_function):
  # get the function to tests
  functions_to_test = inspect.getmembers(Functions, predicate=inspect.isclass)
  print(f"{len(functions_to_test)} HydrOpTop functions to test")
  
  #create PFLOTRAN simulation object
  print("Create and run PFLOTRAN simulation to create functions inputs")
  pflotranin = "../PFLOTRAN_problems/quad_128_hetero/pflotran.in"
  sim = PFLOTRAN(pflotranin)
  #create a random material parameter
  p = np.random.random(sim.get_grid_size())
  perm_data = np.genfromtxt("../PFLOTRAN_problems/quad_128_hetero/permeability_field.csv",
                             comments='#')
  cell_ids, perm_field = perm_data[:,0], perm_data[:,1]
  sim.create_cell_indexed_dataset(perm_field, "permeability", 
                                  "permeability.h5", cell_ids)
  
  sim.run_PFLOTRAN()
  
  err = False
  #Test function
  for name,function in functions_to_test:
    print(f"\nTest HydrOpTop function \"{name}\":")
    #try:
    #create objective
    obj = function()
    #set objective inputs
    inputs = []
    for output in obj.__get_PFLOTRAN_output_variable_needed__():
      if output == "CONNECTION_IDS": 
        inputs.append(sim.get_internal_connections())
        continue
      inputs.append(sim.get_output_variable(output))
    obj.set_inputs(inputs)
    obj.set_p_to_cell_ids(np.arange(1,sim.n_cells+1))
    #compare
    ret_code = debug_function(obj, p)
    if ret_code: err = True
    #except:
    #  print("Error occuring during function evaluation")
    #  err = True
  return err

  
