import sys
import os
path = os.getcwd() + '/../'
sys.path.append(path)

import pflotran_utils as pu
import numpy as np
import h5py

import nlopt
                                  
from HydrOpTop.Objectives import Sum_Liquid_Piezometric_Head
from HydrOpTop.Constrains import Constrains.Maximum_Volume
from HydrOpTop.Materials import Permeability
from HydrOpTop import PFLOTRAN
from HydrOpTop import Craft


if __name__ == "__main__":
  #create PFLOTRAN simulation object
  pflotranin = "pflotran.in"
  sim = PFLOTRAN(pflotranin)
  
  #get cell ids in the region to optimize and parametrize permeability
  #same name than in pflotran input file
  pit_ids = sim.get_region_ids("pit")
  perm = Permeability(pit_ids, [1e-14, 1e-10])
  
  #define cost function as sum of the head in the pit
  cf = Sum_Liquid_Piezometric_Head(ids_to_sum=pit_ids, penalizing_power=1)
  
  #define maximum volume constrains
  max_vol = Maximum_Volume(pit_ids, 0.4)
  
  #craft optimization problem
  #i.e. create function to optimize, initiate IO array in classes...
  crafted_problem = Craft(cf, sim, [perm], [max_vol])
  
  
  
  #initialize optimizer
  algorithm = nlopt.LD_MMA
  opt = nlopt.opt(algorithm, crafted_problem.problem_size())
  opt.set_min_objective(crafted_problem.nlopt_function_to_optimize)
  
  #define constrain
  opt.add_inequality_constraint(max_vol, 0.001) #function, tolerance
  opt.set_lower_bounds(np.zeros(crafted_problem.problem_size(), dtype='f8')+0.01)
  opt.set_upper_bounds(np.ones(crafted_problem.problem_size(), dtype='f8'))
  
  #define stop criterion
  opt.set_ftol_rel(0.00001)
  
  #initial guess
  p = np.zeros(n_to_opt,dtype='f8')
  p[:] = 0.01
  p[:190] = 1.
  
  p_opt = opt.optimize(p)
  
