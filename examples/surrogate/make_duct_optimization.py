"""
Duct flow
=========
"""

# %% 
# This example briefly demonstrates how Darcy flow can be used as a surrogate model of more complex Navier-Stokes problem as in Zhao et al. (2018). https://doi.org/10.1016/j.ijheatmasstransfer.2017.09.090
# 

import numpy as np
                                  
from HydrOpTop.Functions import Sum_Variable #objective
from HydrOpTop.Functions import Volume_Percentage #constrain
from HydrOpTop.Materials import Log_SIMP
from HydrOpTop.Crafter import Steady_State_Crafter
from HydrOpTop.Filters import Density_Filter
from HydrOpTop.Solvers import PFLOTRAN

if __name__ == "__main__":
    #create PFLOTRAN simulation object
    pflotranin = "pflotran.in"
    sim = PFLOTRAN(pflotranin)
    inlet_vol = sim.get_region_ids("inlet_vol")

    #get cell ids in the region to optimize and parametrize permeability
    #same name than in pflotran input file
    perm = Log_SIMP(cell_ids_to_parametrize="all", property_name="PERMEABILITY", bounds=[1e-10, 1e-7], power=3)

    #define cost function as sum of the head in the pit
    cf = Sum_Variable("LIQUID_PRESSURE", ids_to_consider=inlet_vol)

    #define maximum volume constrains
    max_vol = (Volume_Percentage("parametrized_cell"), '<', 0.1)

    filter_ = Density_Filter(0.05)

    #craft optimization problem
    #i.e. create function to optimize, initiate IO array in classes...
    crafted_problem = Steady_State_Crafter(cf, sim, [perm], [max_vol], [filter_])
    crafted_problem.IO.output_every_iteration(5)
    crafted_problem.IO.define_output_format("vtu")

    #initialize optimizer
    p = np.zeros(crafted_problem.get_problem_size(),dtype='f8') + 0.09
    p_opt = crafted_problem.optimize(optimizer="nlopt-mma", action="minimize", max_it=75, initial_guess=p)
