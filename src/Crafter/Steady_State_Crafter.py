import h5py
import numpy as np
import nlopt

from HydrOpTop.Adjoints import Sensitivity_Richards


class Steady_State_Crafter:
  """
  Craft a topology optimization problem in steady state
  Argument:
  - mat_props: a list of material properties that vary with the density
               parameter p (Material classes instances)
  - solver: object that manage the PDE solver (PFLOTRAN class instance)
  - objectif: the objective function (Objectif class instance)
  - constrains: a list of constrains (Constrain class instances
  - coupling: specify how each material properties should be optimized
                       (coupled = one unique p parameter per cell, half = coupled
                       for duplicate ids to optimize in each material, none =
                       each material properties have a separate parameter) 
                       (default=total)
  """
  def __init__(self, mat_props, solver, objectif, constrains, coupling="total"):
    self.mat_props = mat_props
    self.solver = solver
    self.obj = objectif
    self.constrains = constrains
    self.coupling = coupling
    
    #self.Xi = None #store material properties
    self.adjoint = None
    self.Yi = None #store PFLOTRAN output
    
    self.__initialize_IO_array__()
    self.__initialize_adjoint__() #TODO complete
    return
  
  def nlopt_function_to_optimize(self, p, grad):      
    ### UPDATE MAT PROPERTIES AND RUN PFLOTRAN ###
    #Given p, update material properties
    for mat_prop in self.mat_props:
      X = mat_prop.convert_p_to_mat_properties(p)
      self.solver.create_cell_indexed_dataset(X, mat_prop.get_name().lower(),
                    X_ids=mat_prop.get_cell_ids_to_parametrize(), resize_to=True)
    #run PFLOTRAN
    self.solver.run_PFLOTRAN()
    
    ### GET PFLOTRAN OUTPUT ###
    for i,var in enumerate(self.objective.__get_PFLOTRAN_output_variable_needed__()):
      self.solver.get_output_variable(var, self.Yi[i], -1) #last timestep
    
    ### EVALUATE COST FUNCTION AND ITS DERIVATIVE ###
    # note that we have in place assignement, so we don't have to
    # update the Yi in the objective
    cf = self.objectif.evaluate()
    if grad.size > 0:
      ### CREATE ADJOINT AT FIRST CALL ###
      if first:
        self.__initialize_adjoint__() #TODO: find a way to remove that
        first = False
      ### UPDATE ADJOINT ###
      #cost derivative to pressure (vector)
      self.obj.d_objectives_dP(self.adjoint.dc_dP)
      #cost derivative to mat properties (vector)
      for i,mat_prop in enumerate(self.mat_props):
        var = mat_prop.get_name()
        if self.obj.__depend_of__(var):
          self.obj.d_objectives_d_inputs(var, self.adjoint.dXi_dp[i].data)
      #update matrix
      #note the I,J do not change, only the data
      #residual according to pressure
      self.solver.update_sensibility("LIQUID_PRESSURE", self.adjoint.dR_dP)
      #residual according to mat_prop
      for i,mat_prop in enumerate(self.mat_props):
        self.solver.update_sensibility(mat_prop.get_name(),
                                       self.adjoint.dR_dXi[i])
      #material property deriv according to mat parameter
      for i,mat_prop in enumerate(self.mat_props):
        mat_prop.d_mat_properties(p, self.adjoint.dXi_dp[i].data)
      
      ### COMPUTE ADJOINT ###
      self.adjoint.compute_sensitivity(grad)
    return cf
    
  
  def __initialize_IO_array__(self):
    #create material parameter p
    if coupling == "total":
      #verify if each ids are the same
      X = self.mat_props[0].get_cell_ids_to_parametrize()
      if len(self.mat_props) > 1:
        for x in self.mat_props:
          if x != X: 
            print("Different cell ids to optimize")
            print("Can not use 'total' coupling method")
            exit(1)
      self.problem_size = len(X)
    #elif coupling == "none":
    #  self.Xi = 
    else:
      print("Error: Other coupling method not yet implemented")
      exit(1)
    
    #initialize output
    n_outputs = len(self.obj.__get_PFLOTRAN_output_variable_needed__())
    self.Yi = [np.zeros(self.solver.n_cells, dtype='f8') for x in range(n_outputs)]
    return
  
  def __initialize_adjoint__(self):
    
    self.adjoint = Sensitivity_Richards()
    return
    

