import sys
import os
path = os.getcwd() + '/../../'
sys.path.append(path)

import numpy as np
import h5py

import nlopt
                                  
from HydrOpTop.Functions import Sum_Flux, Volume_Percentage
from HydrOpTop.Materials import Permeability
from HydrOpTop.Crafter import Steady_State_Crafter
from HydrOpTop.Filters import Density_Filter
from HydrOpTop import PFLOTRAN
from HydrOpTop import IO



if __name__ == "__main__":
  ###
  # Create HydrOpTop optimization problem
  ###
  pflotranin = "pflotran.in" #path to the PFLOTRAN input file
  sim = PFLOTRAN(pflotranin, mesh_info='mesh_info.h5') #add mesh_info argument to reduce IO
  sim.set_parallel_calling_command(2,"mpiexec.mpich") #call PFLOTRAN using 2 processes
  
  #get cell ids in the region to optimize (same name than in pflotran input file)
  pit_ids = sim.get_region_ids("Pit") #get the ids corresponding to the excavation
  #create the SIMP parametrization
  perm = Permeability([5e-12,1e-10], cell_ids_to_parametrize=pit_ids, power=3, log=False) 
  
  #define cost function
  barrier_connections = sim.get_connections_ids_integral_flux("barrier")
  cf = Sum_Flux(barrier_connections, option="absolute")
  
  #define volume percentage constrain
  max_vol = Volume_Percentage(pit_ids, 0.15)
  
  #define filter to impose a minimum length
  filter = Density_Filter(3)
  
  #craft optimization problem
  #i.e. create function to optimize, initiate IO array in classes...
  crafted_problem = Steady_State_Crafter(cf, sim, [perm], [max_vol], filter)
  crafted_problem.IO.output_every_iteration(2)
  crafted_problem.IO.output_gradient(True)
  crafted_problem.IO.define_output_format('vtu')
  
  ###
  # At this point, the optimization problem is set
  # Create the optimizer using nlopt
  ###
  algorithm = nlopt.LD_MMA #use MMA algorithm
  opt = nlopt.opt(algorithm, crafted_problem.get_problem_size()) #set up optimization
  opt.set_max_objective(crafted_problem.nlopt_function_to_optimize) #pass the cost function to nlopt
  
  #add constrains
  opt.add_inequality_constraint(crafted_problem.nlopt_constrain(0), #max volume
                                0.005) #function, tolerance
  
  #define minimum and maximum bounds for the optimization variable (i.e. p)
  opt.set_lower_bounds(np.zeros(crafted_problem.get_problem_size(), dtype='f8')+0.001)
  opt.set_upper_bounds(np.ones(crafted_problem.get_problem_size(), dtype='f8'))
  
  #define stop criterion
  opt.set_maxeval(30)
  
  ###
  # Perform optimization
  ###
  p = np.zeros(crafted_problem.get_problem_size(), dtype='f8')+0.05 #initial guess
  try:
    p_opt = opt.optimize(p)
    crafted_problem.output_results()
  except(KeyboardInterrupt):
    crafted_problem.output_results()
  
