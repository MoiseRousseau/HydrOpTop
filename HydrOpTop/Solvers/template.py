import numpy as np
from scipy.sparse import coo_matrix, dia_matrix
import h5py
import subprocess
import os
import time


class PFLOTRAN:
  def __init__(self):
    self.solver_command = ""
    self.log = "solver_log.log" #log file where the solver verbose is redirected
    self.no_run = False
    return
    
  def get_grid_size(self):
    return self.n_cells
  
  def disable_run(self):
    """
    Disable running simulation (for debug purpose)
    """
    self.no_run = True
    return
  
  def get_var_location(self):
    return "cell" #or "points"
  
  
  def create_cell_indexed_dataset(self, X_dataset, outfile="",
                                        X_ids=None, resize_to=True):
    """
    Write a variable given by the Crafter to be read by the solver
    Arguments:
    - X_dataset: the dataset
    - X_ids: the cell ids matching the dataset value in X
             (i.e. if X_ids = [5, 3] and X_dataset = [1e7, 1e8],
             therefore, cell id 5 will have a X of 1e7 and 3 with 1e8).
             By default, assumed in natural order
    - resize_to: boolean for resizing the given dataset to number of cell
                 (default = True)
    """
    return
   
    
  def get_mesh(self):
    """
    Return the mesh in meshio format
    """
    return vert, cells, indexes

  
  # running
  def run(self):
    """
    Run the solver
    """
    if self.no_run: return 0
    print("Running Solver: ",end='')
    cmd = [self.solver_command, self.input_file, self.outputfile] #example of a solver which need input file and output file
    tstart = time.time()
    ret = subprocess.call(cmd, stdout=open(self.log,'w'))
    print(f"{time.time() - tstart} s to run simulation")
    if ret: 
      print("\n!!! Error occured in Solver !!!")
      print("Please see " + self.log)
    return ret
  
  
  
  def get_output_variable(self, var, out=None, i_timestep=-1):
    """
    Return output variable after simulation
    If out array is provided, copy variable to array, else, create a new one
    Arguments:
    - var: the variable name as in PFLOTRAN input file under VARIABLES block
    - out: the numpy output array (default=None)
    - timestep: the i-th timestep to extract
    """
    return out
  
  def get_sensitivity(self, var, timestep=None, coo_mat=None):
    """
    Return a scipy COO matrix representing the derivative
    of the residual according to the given variable. 
    - var: the input variable to get sensitivity
    - timestep: the timestep 
    - coo_mat: a scipy COO matrix for in place assignment
    """
    if coo_mat is None:
      new_mat = coo_matrix( (data,(i.astype('i8')-1,j.astype('i8')-1)), dtype='f8')
      return new_mat
    else:
      coo_mat.data[:] = data
    return
    
    
