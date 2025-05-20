"""
2D pervious surround
====================
"""

import numpy as np
                 
from HydrOpTop.Functions import Head_Gradient, Volume_Percentage
from HydrOpTop.Materials import Log_SIMP
from HydrOpTop.Crafter import Steady_State_Crafter
from HydrOpTop.Solvers import PFLOTRAN


if __name__ == "__main__":
  #create PFLOTRAN simulation object
  pflotranin = "pflotran_perv_surr_2D.in"
  sim = PFLOTRAN(pflotranin)
  
  #get cell ids in the region to optimize and parametrize permeability
  #same name than in pflotran input file
  pit_ids = sim.get_region_ids("pit")
  perm = Log_SIMP(cell_ids_to_parametrize=pit_ids, property_name="PERMEABILITY", bounds=[1e-14, 1e-10], power=3)
  
  #define cost function as sum of the head in the pit
  cf = Head_Gradient(pit_ids, power=1)
  
  #define maximum volume constrains
  max_vol = (Volume_Percentage(pit_ids), '<', 0.2)
  
  #craft optimization problem
  #i.e. create function to optimize, initiate IO array in classes...
  crafted_problem = Steady_State_Crafter(cf, sim, [perm], [max_vol])
  crafted_problem.IO.output_every_iteration(10)
  crafted_problem.IO.define_output_format("vtu")
  
  #initialize optimizer
  p = np.zeros(crafted_problem.get_problem_size(),dtype='f8') + 0.1
  crafted_problem.optimize(optimizer="nlopt-mma", action="minimize", max_it=50, initial_guess=p)
  
