#First, import necessary libraries
import sys
import os
path = os.getcwd() + '/../../'
sys.path.append(path)

import numpy as np
import time
                                  
from HydrOpTop.Functions import Mechanical_Compliance, Volume_Percentage
from HydrOpTop.Materials import SIMP
from HydrOpTop.Filters import Density_Filter
from HydrOpTop.Crafter import Steady_State_Crafter
from HydrOpTop.Solvers import Linear_Elasticity_2D


if __name__ == "__main__":
  #Create a timer to time the optimization
  t = time.time()
  
  #Create the simulation object with the linear elasticity solver shield
  sim = Linear_Elasticity_2D("cantilever")
  
  #Parametrize the cell of the whole mesh using a SIMP parametrization of the Young modulus between 0 (p=0) and 2000 MPa (p=1)
  young_modulus = SIMP(cell_ids_to_parametrize="all", property_name="YOUNG_MODULUS", bounds=[0, 2000], power=3)
  
  #Define the cost function (mechanical compliance) defined at all the nodes of the simulation
  cf = Mechanical_Compliance(ids_to_consider="everywhere")
  
  #Define the maximum volume constraint
  max_vol = Volume_Percentage("parametrized_cell")
  max_vol.constraint_tol = 0.5 #max volume percentage of 50%
  
  #Create the density filter using a ball radius of 0.3 units to avoid checkerboard effect
  dfilter = Density_Filter(0.3)
  
  #Craft the optimization problem
  #i.e. create function to optimize, initiate IO array in classes...
  crafted_problem = Steady_State_Crafter(objective=cf, 
                                         solver=sim, 
                                         mat_props=[young_modulus], 
                                         constraints=[max_vol], 
                                         filters=[dfilter])
  
  #Define the output behavior (output density parameter every 2 iterations in vtu format)
  crafted_problem.IO.output_every_iteration(2)
  crafted_problem.IO.define_output_format("vtu")
  
  #Create a initial guess for the optimization
  p_ini = np.zeros(crafted_problem.get_problem_size(),dtype='f8') + 0.2
  
  #Use the MMA algorithm from the library ``nlopt`` to minimize the cost function, until a maximum of 50 iterations is reached or when the relative variation of the cost function is below 0.0001
  out = crafted_problem.optimize(optimizer="nlopt-mma", action="minimize", max_it=50, ftol=0.0001, initial_guess=p_ini)
  
  #Output the final optimized and filtered density parameter in a out.vtu file
  crafted_problem.IO.write_fields_to_file([out.p_opt_filtered], "./out.vtu", ["Filtered_density"])
  
  print(f"Elapsed time: {time.time()-t} seconds")
  
