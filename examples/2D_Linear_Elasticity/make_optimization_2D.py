import sys
import os
path = os.getcwd() + '/../../'
sys.path.append(path)

import numpy as np
                                  
from HydrOpTop.Functions import Mechanical_Compliance, Volume_Percentage
from HydrOpTop.Materials import SIMP
from HydrOpTop.Filters import Density_Filter, Volume_Preserving_Heavyside_Filter
from HydrOpTop.Crafter import Steady_State_Crafter
from HydrOpTop.Solvers import Linear_Elasticity_2D


if __name__ == "__main__":
  #create PFLOTRAN simulation object
  sim = Linear_Elasticity_2D("cantilever")
  
  #get cell ids in the region to optimize and parametrize permeability
  #same name than in pflotran input file
  perm = SIMP(cell_ids_to_parametrize="all", property_name="YOUNG_MODULUS", bounds=[0, 2000], power=3)
  
  #define cost function
  cf = Mechanical_Compliance(ids_to_consider="everywhere")
  
  #define maximum volume constrains
  max_vol = Volume_Percentage("parametrized_cell", 0.5)
  
  #define filter
  dfilter = Density_Filter(0.3)
  hfilter = Volume_Preserving_Heavyside_Filter(0.5, 1, max_vol)
  
  #craft optimization problem
  #i.e. create function to optimize, initiate IO array in classes...
  crafted_problem = Steady_State_Crafter(cf, sim, [perm], [max_vol], filters=[dfilter, hfilter])
  crafted_problem.IO.output_every_iteration(2)
  crafted_problem.IO.define_output_format("vtu")
  
  #initialize optimizer
  p = np.zeros(crafted_problem.get_problem_size(),dtype='f8') + 0.2
  
  #optimize in several pass to reach discrete distribution
  p_opt = crafted_problem.optimize(optimizer="nlopt-mma", action="minimize", max_it=50, ftol=0.0001, initial_guess=p)
  hfilter.update_stepness(2)
  p_opt[:] = crafted_problem.optimize(optimizer="nlopt-mma", action="minimize", max_it=20, initial_guess=p_opt)
  hfilter.update_stepness(4)
  p_opt[:] = crafted_problem.optimize(optimizer="nlopt-mma", action="minimize", max_it=10, initial_guess=p_opt)
  hfilter.update_stepness(8)
  p_opt[:] = crafted_problem.optimize(optimizer="nlopt-mma", action="minimize", max_it=10, initial_guess=p_opt)
  hfilter.update_stepness(20)
  crafted_problem.optimize(optimizer="nlopt-mma", action="minimize", max_it=5, initial_guess=p_opt)
  
