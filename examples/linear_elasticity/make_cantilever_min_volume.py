"""
Minimize volume with maximum compliance
#######################################
"""

# %%
# This is a variation of the standart design optimisation problem where the minimum volume structure is seeked given a maximum compliance.
# It illustrate the modularity of how Functions is implemented in HydrOpTop and can be used either as a constrain or a main objective.

import numpy as np

from mechanical_compliance import Mechanical_Compliance
from HydrOpTop.Functions import Volume_Percentage
from HydrOpTop.Materials import SIMP
from HydrOpTop.Filters import Density_Filter, Heaviside_Filter
from HydrOpTop.Crafter import Steady_State_Crafter
from linear_elasticity_2d import Linear_Elasticity_2D


if __name__ == "__main__":
    #create solver simulation object
    sim = Linear_Elasticity_2D("cantilever")
    all_cells = sim.get_region_ids("__all__")
    all_nodes = sim.get_node_region_ids("__all__")

    #get cell ids in the region to optimize and parametrize permeability
    young_modulus = SIMP(
        cell_ids_to_parametrize=all_cells,
        property_name="YOUNG_MODULUS",
        bounds=[0, 2000], power=3
    )

    #define cost function
    max_vol = Volume_Percentage(all_cells)

    #define maximum compliance
    MC = (Mechanical_Compliance(ids_to_consider=all_nodes), '<', 5e-2)

    #define filter
    dfilter = Density_Filter(all_cells, 0.3)
    hfilter = Heaviside_Filter(all_cells, 0.5, 1)

    #craft optimization problem
    #i.e. create function to optimize, initiate IO array in classes...
    crafted_problem = Steady_State_Crafter(
        max_vol, sim, [young_modulus], [MC], filters=[dfilter, hfilter],
        deriv="adjoint", deriv_args={"method":"direct"},
    )
    crafted_problem.IO.output_every_iteration(2)
    crafted_problem.IO.define_output_format("vtu")

    #initialize optimizer
    p_ini = np.ones(crafted_problem.get_problem_size(),dtype='f8')

    #optimize in several pass to reach discrete distribution
    out = crafted_problem.optimize(
        optimizer="nlopt-ccsaq", action="minimize", initial_guess=p_ini,
        optimizer_args={"set_maxeval":30, "set_ftol_rel":1e-6, "set_initial_step":10.},
    )

    for stepness in [2,4,8,20]:
        print("Increase stepness to", stepness)
        hfilter.stepness = stepness
        out = crafted_problem.optimize(
            optimizer="nlopt-ccsaq", action="minimize", initial_guess=out.p_opt,
            optimizer_args={"set_maxeval":20, "set_ftol_rel":1e-6, "set_initial_step":3.},
        )

    crafted_problem.IO.write_fields_to_file([out.p_opt_filtered], "./out.vtu", ["Filtered_density"])
