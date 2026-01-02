"""
Ressource maximization with drawdown constrain
##############################################

HydrOpTop can be used to maximize the area of an excavation relative to a given objective, for example maximizing the excavation of a ressource submitted to some ecological constrains.
"""

import numpy as np
import pandas as pd
                                  
from HydrOpTop.Functions import Drawdown, p_Weighted_Sum
from HydrOpTop.Materials import Log_SIMP
from HydrOpTop.Crafter import Steady_State_Crafter
from HydrOpTop.Filters import Density_Filter, Volume_Preserving_Heaviside_Filter
from HydrOpTop.Solvers import PFLOTRAN

if __name__ == "__main__":
    # Create PFLOTRAN simulation object
    pflotranin = "excavated.in"
    sim = PFLOTRAN(pflotranin)

    # Create objective function
    # Load a CSV file with the ratio of ressource per volume and the corresponding ids
    resources = pd.read_csv("./resources.csv",index_col=0)
    pit_ids = resources.index.values
    cf = p_Weighted_Sum(
        field=resources["Resources"].values,
        field_ids=pit_ids,
    )

    # Create 1 constrain per each maximum drawdown ids
    max_drawdown_ids = [1135,984] #impose max drawdown on cell id 984 and 1135
    initial_head = [230.1,234.5]
    max_drawdown = 12.
    constrains = [
        (Drawdown(h, c),'<',max_drawdown) for h,c in zip(initial_head,max_drawdown_ids)
    ]

    # Parametrization
    perm = Log_SIMP(
        cell_ids_to_parametrize=pit_ids,
        property_name="PERMEABILITY",
        bounds=[1e-13, 1e-4],
        power=1
    )

    # Filter
    dfilter = Density_Filter(pit_ids, radius=8.)
    hfilter = Volume_Preserving_Heaviside_Filter(pit_ids, 0.5, 4)

    #craft optimization problem
    #i.e. create function to optimize, initiate IO array in classes...
    crafted_problem = Steady_State_Crafter(
        cf, sim, [perm], constrains, [dfilter,hfilter],
        deriv="adjoint", deriv_args={"method":"direct"}
    )
    crafted_problem.IO.output_every_iteration(1)
    crafted_problem.IO.output_material_properties()
    crafted_problem.IO.define_output_format("vtu")

    #initialize optimizer
    p = np.zeros(crafted_problem.get_problem_size()) + 0.01
    res = crafted_problem.optimize(
        optimizer="nlopt-ccsaq", action="maximize", max_it=15, stop={"ftol":0.0001}, initial_guess=p
    )


    dfilter.write_vtu_fitered_density("debug.vtp", crafted_problem.filter_sequence.apply_filters(res.p_opt,0))

    crafted_problem.IO.plot_convergence_history(include_constraints=True)
    
