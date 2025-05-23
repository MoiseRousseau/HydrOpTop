"""
Drains placement
################

Some engineers already faced the problem to efficiently design drains to lower the groundwater table for example.
HydrOpTop can be used to assess what the optimal drain pattern given a volume constrain as follow.

In the same category than https://onlinelibrary.wiley.com/doi/abs/10.1002/nme.1811
"""

import numpy as np
                                  
from HydrOpTop.Functions import Mean_Liquid_Piezometric_Head #objective
from HydrOpTop.Functions import Volume_Percentage #constrain
from HydrOpTop.Materials import Log_SIMP
from HydrOpTop.Crafter import Steady_State_Crafter
from HydrOpTop.Filters import Density_Filter
from HydrOpTop.Solvers import PFLOTRAN


if __name__ == "__main__":
  #create PFLOTRAN simulation object
  pflotranin = "drainage.in"
  sim = PFLOTRAN(pflotranin)
  
  #get cell ids in the region to optimize and parametrize permeability
  #same name than in pflotran input file
  perm = Log_SIMP(cell_ids_to_parametrize="all", property_name="PERMEABILITY", bounds=[1e-14, 1e-10], power=3)
  
  #define cost function as sum of the head in the pit
  cf = Mean_Liquid_Piezometric_Head(ids_to_sum="everywhere", penalizing_power=1)
  
  #define maximum volume constrains
  max_vol = (Volume_Percentage("parametrized_cell"), '<', 0.2)
  
  filter_ = Density_Filter(10.)
  
  #craft optimization problem
  #i.e. create function to optimize, initiate IO array in classes...
  crafted_problem = Steady_State_Crafter(cf, sim, [perm], [max_vol], [filter_])
  crafted_problem.IO.output_every_iteration(1)
  crafted_problem.IO.output_gradient()
  crafted_problem.IO.output_material_properties()
  crafted_problem.IO.define_output_format("vtu")
  
  #initialize optimizer
  p = np.zeros(crafted_problem.get_problem_size(),dtype='f8') + 0.15
  p_opt = crafted_problem.optimize(optimizer="nlopt-mma", action="minimize", max_it=75, ftol=0.0001, initial_guess=p)
    
