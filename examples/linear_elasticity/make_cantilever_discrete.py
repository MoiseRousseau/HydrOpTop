"""
Cantilever discrete design
##########################

This third example is a new variation of the cantilever design problem with a three fields optimization, i.e. with the raw, the filtered and the projected density parameter fields.
Projected density parameter field is obtained by projecting the filtered field using a smooth Heaviside function.
This strategy could be use to obtain discrete 0-1 design at the end of the optimization by carring a multi-steps optimization process and increasing the steepness of the smooth Heaviside function at each step.
We first carry out the optimization problem as in the cantilever example
"""

import numpy as np

from mechanical_compliance import Mechanical_Compliance
from HydrOpTop.Functions import Volume_Percentage
from HydrOpTop.Materials import SIMP
from HydrOpTop.Filters import Density_Filter, Heaviside_Filter
from HydrOpTop.Crafter import Steady_State_Crafter
from linear_elasticity_2d import Linear_Elasticity_2D

if __name__ == "__main__":
    #create PFLOTRAN simulation object
    sim = Linear_Elasticity_2D("cantilever")
    all_cells = sim.get_region_ids("__all__")
    all_nodes = sim.get_node_region_ids("__all__")

    #get cell ids in the region to optimize and parametrize permeability
    #same name than in pflotran input file
    young_modulus = SIMP(
        cell_ids_to_parametrize=all_cells,
        property_name="YOUNG_MODULUS",
        bounds=[0, 2000], power=3
    )

    #define cost function
    cf = Mechanical_Compliance(ids_to_consider=all_nodes)

    #define maximum volume constrain
    max_vol = (Volume_Percentage(all_cells), '<', 0.5)

    #define filter
    dfilter = Density_Filter(all_cells, 0.3)
    hfilter = Heaviside_Filter(all_cells, 0.5, 1)

    #craft optimization problem
    #i.e. create function to optimize, initiate IO array in classes...
    crafted_problem = Steady_State_Crafter(
        cf, sim, [young_modulus], [max_vol], filters=[dfilter, hfilter],
        deriv="adjoint", deriv_args={"method":"direct"},
    ) #apply first density filter (dfilter) and then the Heaviside filter (hfilter)
    crafted_problem.IO.output_every_iteration(2)
    crafted_problem.IO.define_output_format("vtu")

    #initialize optimizer
    p = np.zeros(crafted_problem.get_problem_size(),dtype='f8') + 0.2

    # %%
    # The optimization is carried out in several step, starting from a smooth Heavyside density filter and increasing the steepness parameter.
    # As this can be difficult to converge for high step parameter, we increase it gradually to to reach a discrete distribution of material.

    out = crafted_problem.optimize(
        optimizer="nlopt-ccsaq", action="minimize", initial_guess=p,
        optimizer_args={"set_maxeval":30, "set_ftol_rel":1e-6, "set_initial_step":1.},
    )

    for stepness in [2,4,8,12]:
        print("Increase stepness to", stepness)
        hfilter.stepness = stepness
        out = crafted_problem.optimize(
            optimizer="nlopt-ccsaq", action="minimize", initial_guess=out.p_opt,
            optimizer_args={"set_maxeval":20, "set_ftol_rel":1e-6, "set_initial_step":3.},
        )

    crafted_problem.IO.write_fields_to_file([out.p_opt_filtered], "./out.vtu", ["Filtered_density"])
    crafted_problem.IO.plot_convergence_history()


