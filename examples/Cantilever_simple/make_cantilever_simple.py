import sys
import os
path = os.getcwd() + '/../../'
sys.path.append(path)
import time

import numpy as np
                                  
from HydrOpTop.Functions import Mechanical_Compliance, Volume_Percentage
from HydrOpTop.Materials import SIMP
from HydrOpTop.Filters import Density_Filter, Volume_Preserving_Heavyside_Filter
from HydrOpTop.Crafter import Steady_State_Crafter
from HydrOpTop.Solvers import Linear_Elasticity_2D


if __name__ == "__main__":
  t = time.time()
  #create solver simulation object
  sim = Linear_Elasticity_2D("cantilever")
  
  #get cell ids in the region to optimize and parametrize permeability
  #same name than in pflotran input file
  young_modulus = SIMP(cell_ids_to_parametrize="all", property_name="YOUNG_MODULUS", bounds=[0, 2000], power=3)
  
  #define cost function
  cf = Mechanical_Compliance(ids_to_consider="everywhere")
  
  #define maximum volume constrains
  max_vol = Volume_Percentage("parametrized_cell", 0.5)
  
  #define filter
  dfilter = Density_Filter(0.3)
  
  #craft optimization problem
  #i.e. create function to optimize, initiate IO array in classes...
  crafted_problem = Steady_State_Crafter(cf, sim, [young_modulus], [max_vol], filters=[dfilter])
  crafted_problem.IO.output_every_iteration(2)
  crafted_problem.IO.define_output_format("vtu")
  
  #initialize optimizer
  p_ini = np.zeros(crafted_problem.get_problem_size(),dtype='f8') + 0.2
  
  #optimize in several pass to reach discrete distribution
  out = crafted_problem.optimize(optimizer="nlopt-mma", action="minimize", max_it=50, ftol=0.0001, initial_guess=p_ini)
  
  crafted_problem.IO.write_fields_to_file([out.p_opt_filtered], "./out.vtu", ["Filtered_density"])
  
  print(f"Elapsed time: {time.time()-t} seconds")
  
