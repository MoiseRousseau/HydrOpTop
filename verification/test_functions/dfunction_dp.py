#
# This script test the derivative of the different objective function
# relative to the material parameter p and compare it to finite difference
# calculation
#

import sys
import os
path = os.getcwd() + '/../../'
sys.path.append(path)

import numpy as np
import h5py
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
  pflotranin = "simulation/a_problem.in"
  sim = PFLOTRAN(pflotranin)
  #create a random material parameter
  p = np.random.random(sim.get_grid_size())
  #create material dataset
  
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
      inputs.append(sim.get_output_variable(output))
    obj.set_inputs(inputs)
    #compare
    ret_code = debug_function(obj, p)
    if ret_code: err = True
    #except:
    #  print("Error occuring during function evaluation")
    #  err = True
  return err



if __name__ == "__main__":
  from HydrOpTop.debug import compare_dfunction_dp_with_FD
  err = common_compare(compare_dfunction_dp_with_FD)
  if err: exit(1)
  else: exit(0)

  
