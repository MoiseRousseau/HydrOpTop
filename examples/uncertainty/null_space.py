"""
Sample different hydraulic conductivity using the Null Space method
-------------------------------------------------------------------

We use a Randomized Maximum Likelihood approach where either the observations
are perturbed (which may be representative of the error in measurement) and
the calibration parameter (in practice, the hydraulic conductivity at the 
pilot points).

Perturbed observations are drawn from a true observation with Gaussian noise
while parameters are drawn within the null-space of the Jacobian.
"""

import numpy as np
import pandas as pd

from HydrOpTop.Solvers import PFLOTRAN
from HydrOpTop.Functions import Least_Square_Calibration #objective
from HydrOpTop.Materials import Log_SIMP
from HydrOpTop.Crafter import Steady_State_Crafter
from HydrOpTop.Filters import Pilot_Points
from HydrOpTop.Uncertainty import Null_Space_Sampler

if __name__ == "__main__":

    # Recreate the calibration problem

    # Create PFLOTRAN simulation object
    pflotranin = "../parametrization/pflotran_pp.in"
    sim = PFLOTRAN(pflotranin)
    all_cells = sim.get_region_ids("__all__")
    perm = Log_SIMP(
        cell_ids_to_parametrize=all_cells,
        property_name="PERMEABILITY",
        bounds=[1e-14, 1e-12], power=1
    )

    # Parametrize with Pilot Points
    control_points = pd.read_csv("control_points.csv")
    p_opt = control_points["p"].values
    ppfilter = Pilot_Points(
        control_points[["X","Y"]],
        parametrized_cells=all_cells,
        interpolator="RBFInterpolator",
    )

    # Get the observation at ids and create base calibration object
    cell_ids = [444, 789, 920, 1030, 1339]
    head = [230, 250, 227, 146, 210]
    cf = Least_Square_Calibration(head, cell_ids)

    # Craft optimization problem
    crafted_problem = Steady_State_Crafter(
        cf, sim, [perm], [], [ppfilter],
        deriv="adjoint", deriv_args={"method":"direct"},
    )
    crafted_problem.IO.no_output_initial()
    crafted_problem.IO.output_every_iteration(3)
    crafted_problem.IO.define_output_format("vtu")

    # Get Jacobian to compute the null-space
    jac = crafted_problem.get_jacobian(p_opt)

    # Perform Randomized Maximum Likelihood with parameter sampling in
    # the null space
    # Create the NS Sampler
    sampler = Null_Space_Sampler(p_opt, jac, sigma=0.2)
    rng = np.random.default_rng()
    head_std = 3 # Assume 3 m variation of the head
    N_realization = 10

    for i in range(N_realization):

        # Pertub observations and update objective
        h_pertub = head + rng.normal(0, head_std, size=len(head))
        cf.ref_head = h_pertub

        # Get a new parameter field from the sampler
        p_pertub = sampler.propose_sample()

        # Make one LS iteration to correct for non linearities
        crafted_problem.IO.define_output_file(f"{i:03}.randomized_field")
        res = crafted_problem.optimize(
            optimizer="scipy-trf", action="minimize",
            initial_guess=p_pertub,
            optimizer_args={"max_nfev":2},
        )

        # Print distance between optimized and true parameter
        print(np.linalg.norm(res.p_opt - p_opt))
    
