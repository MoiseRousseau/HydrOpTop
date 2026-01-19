"""
2D pervious surround
====================
"""

import numpy as np

from HydrOpTop.Functions import p_Weighted_Cell_Gradient, Volume_Percentage
from HydrOpTop.Filters import Density_Filter
from HydrOpTop.Materials import Log_SIMP
from HydrOpTop.Crafter import Steady_State_Crafter
from HydrOpTop.Solvers import PFLOTRAN


if __name__ == "__main__":
  #create PFLOTRAN simulation object
  pflotranin = "pflotran_perv_surr_2D.in"
  sim = PFLOTRAN(pflotranin)

  #get cell ids in the region to optimize and parametrize permeability
  #same name than in pflotran input file
  pit_ids = sim.get_region_ids("Pit")
  perm = Log_SIMP(cell_ids_to_parametrize=pit_ids, property_name="PERMEABILITY", bounds=[1e-14, 1e-10], power=3)

  #define cost function as sum of the head in the pit
  cf = p_Weighted_Cell_Gradient(pit_ids, variable="LIQUID_HEAD", power=1, invert_weighting=True)

  #define maximum volume constrains
  max_vol = (Volume_Percentage(pit_ids), '<', 0.2)

  # filter
  filter = Density_Filter(pit_ids, radius=12.)

  #craft optimization problem
  #i.e. create function to optimize, initiate IO array in classes...
  crafted_problem = Steady_State_Crafter(
    cf, sim, [perm], [max_vol], [filter],
    deriv="adjoint", deriv_args={"method":"direct"}
  )
  crafted_problem.IO.output_every_iteration(5)
  crafted_problem.IO.define_output_format("vtu")
  crafted_problem.IO.output_gradient()

  #initialize optimizer
  p = np.zeros(crafted_problem.get_problem_size(),dtype='f8') + 0.01
  crafted_problem.optimize(
    optimizer="nlopt-ccsaq", action="minimize", initial_guess=p,
    optimizer_args={"set_maxeval":30, "set_initial_step":100., 'set_ftol_rel':1e-16},
  )

