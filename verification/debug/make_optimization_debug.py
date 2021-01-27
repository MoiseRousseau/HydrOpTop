import sys
import os
path = os.getcwd() + '/../../'
sys.path.append(path)

import numpy as np
import h5py

import nlopt
                                  
from HydrOpTop.Functions import p_Weighted_Sum_Flux
from HydrOpTop.Functions import Volume_Percentage
from HydrOpTop.Materials import Permeability
from HydrOpTop.Crafter import Steady_State_Crafter
from HydrOpTop import PFLOTRAN

from HydrOpTop.debug import compare_adjoint_with_FD, compare_dfunction_dinputs_with_FD, \
                            compare_dfunction_dpressure_with_FD, compare_dfunction_dp_with_FD


if __name__ == "__main__":
  #create PFLOTRAN simulation object
  pflotranin = "../PFLOTRAN_problems/2x2x2/pflotran.in"
  sim = PFLOTRAN(pflotranin)
  
  #get cell ids in the region to optimize and parametrize permeability
  #same name than in pflotran input file
  pit_ids = np.array([1,2,3]) #fail with compare_dfunction_dinputs_with_FD
  #pit_ids = np.array([1,2,3,4,5])
  perm = Permeability([1e-14, 1e-10], cell_ids_to_parametrize=pit_ids, power=3)
  
  #define cost function as sum of the head in the pit
  cf = p_Weighted_Sum_Flux(pit_ids)
  
  #define maximum volume constrains
  max_vol = Volume_Percentage(pit_ids, 0.2)
  
  #craft optimization problem
  #i.e. create function to optimize, initiate IO array in classes...
  crafted_problem = Steady_State_Crafter(cf, sim, [perm], [max_vol])
  crafted_problem.set_adjoint_problem_algo("spsolve")
  
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
  opt.set_maxeval(50)
  
  #initial guess
  p = np.zeros(crafted_problem.get_problem_size(),dtype='f8') + 0.001
  p[:] = np.random.random(crafted_problem.get_problem_size())
  
  #f = h5py.File("p_opt.h5",'r')
  #p = np.array(f["Density parameter optimized"])
  #f.close()
  compare_adjoint_with_FD(crafted_problem, p, [], pertub=1e-2)
  compare_dfunction_dinputs_with_FD(cf,p,cell_to_check=[3,1,2],pertub=1e-6,detailed_info=True)
  exit()
  
  try:
    p_opt = opt.optimize(p)
    sim.create_cell_indexed_dataset(p_opt, "Density parameter optimized","p_opt.h5")
  except(KeyboardInterrupt):
    sim.create_cell_indexed_dataset(crafted_problem.last_p, 
                                    "Density parameter optimized",
                                    f"p_{crafted_problem.func_eval}.h5")
  
