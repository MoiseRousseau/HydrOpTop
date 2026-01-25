"""
2D permeability field calibration using Pilot Points
----------------------------------------------------
"""

import numpy as np

from HydrOpTop.Solvers import PFLOTRAN
from HydrOpTop.Functions import Least_Square_Calibration #objective
from HydrOpTop.Functions import Volume_Percentage #constrain
from HydrOpTop.Materials import Log_SIMP
from HydrOpTop.Crafter import Steady_State_Crafter
from HydrOpTop.Filters import Pilot_Points

if __name__ == "__main__":

    #create PFLOTRAN simulation object
    pflotranin = "pflotran_pp.in"
    sim = PFLOTRAN(pflotranin)
    all_cells = sim.get_region_ids("__all__")
    #get cell ids in the region to optimize and parametrize permeability
    #same name than in pflotran input file
    perm = Log_SIMP(cell_ids_to_parametrize=all_cells, property_name="PERMEABILITY", bounds=[1e-14, 1e-12], power=1)

    #define cost function as sum of the head in the pit
    cell_ids = [444, 789, 920, 1030, 1339]
    head = [230, 250, 227, 146, 210]
    cf = Least_Square_Calibration(head, cell_ids)

    # Parametrize with Pilot Points
    rng = np.random.default_rng(seed=1234) # for reproductibility
    control_points = rng.random((30,2))*200-100
    ppfilter = Pilot_Points(
        control_points,
        parametrized_cells=all_cells,
        interpolator="RBFInterpolator",
    )

    #craft optimization problem
    #i.e. create function to optimize, initiate IO array in classes...
    crafted_problem = Steady_State_Crafter(
        cf, sim, [perm], [], [ppfilter],
        deriv="adjoint", deriv_args={"method":"direct"},
    )
    crafted_problem.IO.output_every_iteration(10)
    crafted_problem.IO.output_gradient()
    crafted_problem.IO.output_material_properties()
    crafted_problem.IO.define_output_format("vtu")

    #initialize optimizer
    p = np.zeros(crafted_problem.get_problem_size()) + 0.5
    #p = 0.999*np.random.rand(crafted_problem.get_problem_size())+0.001 
    p_opt = crafted_problem.optimize(
        optimizer="scipy-trf", action="minimize",
        initial_guess=p,
        optimizer_args={"max_nfev":10},
    )

    crafted_problem.IO.plot_convergence_history()
    
