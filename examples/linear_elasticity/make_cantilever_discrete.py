"""
Cantilever discrete design
##########################

This third example is a new variation of the cantilever design problem with a three fields optimization, i.e. with the raw, the filtered and the projected density parameter fields.
Projected density parameter field is obtained by projecting the filtered field using a smooth Heaviside function.
This strategy could be use to obtain discrete 0-1 design at the end of the optimization by carring a multi-steps optimization process and increasing the steepness of the smooth Heaviside function at each step.
We first carry out the optimization problem as in the cantilever example
"""

import numpy as np

from HydrOpTop.Functions import Volume_Percentage, Mechanical_Compliance
from HydrOpTop.Materials import SIMP
from HydrOpTop.Filters import Density_Filter, Volume_Preserving_Heaviside_Filter
from HydrOpTop.Crafter import Steady_State_Crafter
from linear_elasticity_2d import Linear_Elasticity_2D

if __name__ == "__main__":
    #create PFLOTRAN simulation object
    sim = Linear_Elasticity_2D("cantilever")

    #get cell ids in the region to optimize and parametrize permeability
    #same name than in pflotran input file
    perm = SIMP(cell_ids_to_parametrize="all", property_name="YOUNG_MODULUS", bounds=[0, 2000], power=3)

    #define cost function
    cf = Mechanical_Compliance(ids_to_consider="everywhere")

    #define maximum volume constrain
    max_vol = (Volume_Percentage("parametrized_cell"), '<', 0.5)

    #define filter
    dfilter = Density_Filter(0.3)
    hfilter = Volume_Preserving_Heaviside_Filter(0.5, 1, max_vol)

    #craft optimization problem
    #i.e. create function to optimize, initiate IO array in classes...
    crafted_problem = Steady_State_Crafter(cf, sim, [perm], [max_vol], filters=[dfilter, hfilter]) #apply first density filter (dfilter) and then the Heaviside filter (hfilter)
    crafted_problem.IO.output_every_iteration(2)
    crafted_problem.IO.define_output_format("vtu")

    #initialize optimizer
    p = np.zeros(crafted_problem.get_problem_size(),dtype='f8') + 0.2

    # %%
    # The optimization is carried out in several step, starting from a smooth Heavyside density filter and increasing the steepness parameter.
    # As this can be difficult to converge for high step parameter, we increase it gradually to to reach a discrete distribution of material.

    p_opt = crafted_problem.optimize(optimizer="nlopt-mma", action="minimize", 
	                             max_it=50, ftol=0.0001, initial_guess=p)
    hfilter.update_stepness(2)
    p_opt = crafted_problem.optimize(optimizer="nlopt-mma", action="minimize", 
	                             max_it=20, initial_guess=p_opt.p_opt)
    hfilter.update_stepness(4)
    p_opt = crafted_problem.optimize(optimizer="nlopt-mma", action="minimize", 
	                             max_it=10, initial_guess=p_opt.p_opt)
    hfilter.update_stepness(8)
    p_opt = crafted_problem.optimize(optimizer="nlopt-mma", action="minimize", 
	                             max_it=10, initial_guess=p_opt.p_opt)
    hfilter.update_stepness(20)
    p_opt = crafted_problem.optimize(optimizer="nlopt-mma", action="minimize", 
	                             max_it=5, initial_guess=p_opt.p_opt)

    crafted_problem.IO.write_fields_to_file([p_opt.p_opt_filtered], "./out.vtu", ["Filtered_density"])
    crafted_problem.IO.plot_convergence_history()
