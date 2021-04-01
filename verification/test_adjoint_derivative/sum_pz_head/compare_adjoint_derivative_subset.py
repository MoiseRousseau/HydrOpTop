import sys
import os
path = os.getcwd() + '/../../../'
sys.path.append(path)

import numpy as np
import h5py

import nlopt
                                  
from HydrOpTop.Functions import Mean_Liquid_Piezometric_Head
from HydrOpTop.Functions import Volume_Percentage
from HydrOpTop.Materials import Permeability
from HydrOpTop.Crafter import Steady_State_Crafter
from HydrOpTop import PFLOTRAN

from HydrOpTop.debug import compare_adjoint_with_FD


if __name__ == "__main__":
  #create PFLOTRAN simulation object
  pflotranin = "./../../PFLOTRAN_problems/pit_voronoi/pflotran.in"
  sim = PFLOTRAN(pflotranin)
  
  #get cell ids in the region to optimize and parametrize permeability
  #same name than in pflotran input file
  pit_ids = sim.get_region_ids("pit")
  perm = Permeability([1e-14, 1e-10], cell_ids_to_parametrize=pit_ids, power=3)
  
  #define cost function as sum of the head in the pit
  cf = Mean_Liquid_Piezometric_Head(ids_to_sum=pit_ids, penalizing_power=3)
  
  #define maximum volume constrains
  max_vol = Volume_Percentage(pit_ids, 0.2)
  
  #craft optimization problem
  #i.e. create function to optimize, initiate IO array in classes...
  crafted_problem = Steady_State_Crafter(cf, sim, [perm], [max_vol])
  
  #initialize optimizer
  algorithm = nlopt.LD_MMA
  opt = nlopt.opt(algorithm, crafted_problem.get_problem_size())
  opt.set_min_objective(crafted_problem.nlopt_function_to_optimize)
  
  #define constrain
  opt.add_inequality_constraint(max_vol.nlopt_optimize, 0.001) #function, tolerance
  opt.set_lower_bounds(np.zeros(crafted_problem.get_problem_size(), dtype='f8')+0.001)
  opt.set_upper_bounds(np.ones(crafted_problem.get_problem_size(), dtype='f8'))
  
  #define stop criterion
  opt.set_ftol_rel(0.000001)
  opt.set_maxeval(20)
  
  #initial guess
  p = np.zeros(crafted_problem.get_problem_size(),dtype='f8') + 0.12
  #p = np.random.random(crafted_problem.get_problem_size())
  #fix the p value because FD derivative and pertubation depend on p
                                    
  err = compare_adjoint_with_FD(crafted_problem,p,[21,154,444,608,879],pertub=1e-3, accept=2e-2)
  print("")
  exit(err)
  
  
