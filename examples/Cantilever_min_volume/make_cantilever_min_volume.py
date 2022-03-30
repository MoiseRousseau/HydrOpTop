import time
import numpy as np
                                  
from HydrOpTop.Functions import Mechanical_Compliance, Volume_Percentage
from HydrOpTop.Materials import SIMP
from HydrOpTop.Filters import Density_Filter, Heaviside_Filter
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
  max_vol = Volume_Percentage("parametrized_cell")
  
  #define maximum compliance
  MC = Mechanical_Compliance(ids_to_consider="everywhere")
  MC.constraint_tol = 5e-2
  
  #define filter
  dfilter = Density_Filter(0.3)
  hfilter = Heaviside_Filter(0.5, 1)
  
  #craft optimization problem
  #i.e. create function to optimize, initiate IO array in classes...
  crafted_problem = Steady_State_Crafter(max_vol, sim, [young_modulus], [MC], filters=[dfilter, hfilter])
  crafted_problem.IO.output_every_iteration(2)
  crafted_problem.IO.define_output_format("vtu")
  
  #initialize optimizer
  p_ini = np.ones(crafted_problem.get_problem_size(),dtype='f8')
  
  #optimize in several pass to reach discrete distribution
  out = crafted_problem.optimize(optimizer="nlopt-mma", action="minimize", max_it=50, ftol=1e-10, initial_guess=p_ini)
  
  hfilter.stepness = 2
  out = crafted_problem.optimize(optimizer="nlopt-mma", action="minimize", 
                                   max_it=20, initial_guess=out.p_opt)
  hfilter.stepness = 4
  out = crafted_problem.optimize(optimizer="nlopt-mma", action="minimize", 
                                   max_it=20, initial_guess=out.p_opt)
  hfilter.stepness = 8
  out = crafted_problem.optimize(optimizer="nlopt-mma", action="minimize", 
                                   max_it=20, initial_guess=out.p_opt)
  hfilter.stepness = 20
  out = crafted_problem.optimize(optimizer="nlopt-mma", action="minimize", 
                                   max_it=30, initial_guess=out.p_opt)
  
  crafted_problem.IO.write_fields_to_file([out.p_opt_filtered], "./out.vtu", ["Filtered_density"])
  
  print(f"Elapsed time: {time.time()-t} seconds")
  
