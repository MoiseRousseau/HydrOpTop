import numpy as np

def compute_sensitivity_adjoint(solver, objective, propname, parametrization, adjoint, p=None, rgn=None):
  #generate random with a given seed
  n = solver.get_grid_size()
  if p is None:
    if rgn is None:
      rgn = np.random.default_rng(0)
    p = rgn.random(n)
  prop = parametrization.convert_p_to_mat_properties(p)
  solver.create_cell_indexed_dataset(prop, propname, propname+'.h5')
  #run model
  solver.run()
  #initiate objective
  inputs = {x:None for x in objective.__get_variables_needed__()}
  solver.get_output_variables(inputs)
  objective.set_inputs(inputs)
  #initiate sensitivity
  solved_var = set(objective.__get_variables_needed__())
  solved_var.intersection_update(solver.solved_variables)
  sens = adjoint(solved_var,
                 [parametrization], solver, np.arange(1,n+1)) 
  #compite
  S_adjoint = sens.compute_sensitivity(p, 
                                       objective.d_objective_dY(prop),
                                       objective.d_objective_dX(prop),
                                       [""]) #p, dc_dYi, dc_dXi, Xi_name):
  return S_adjoint


def compute_sensitivity_finite_difference(solver, objective, propname, parametrization, cell_ids_to_test=None, p=None, rgn=None, pertub=1e-3):
  #generate random with a given seed
  n = solver.get_grid_size()
  if p is None:
    if rgn is None:
      rgn = np.random.default_rng(0)
    p = rgn.random(n)
  prop = parametrization.convert_p_to_mat_properties(p)
  solver.create_cell_indexed_dataset(prop, propname, propname+'.h5')
  #run model for current objective
  solver.run()
  #initiate objective
  inputs = []
  for var in objective.__get_variables_needed__():
    inputs.append(solver.get_output_variable(var))
  objective.set_inputs(inputs)
  
  #run finite difference
  ref_obj = objective.evaluate(p)
  if cell_ids_to_test is None: 
    cell_ids_to_test = np.arange(1,n+1)
  deriv = np.zeros(len(cell_ids_to_test),dtype='f8')
  for i,cell in enumerate(cell_ids_to_test):
    print(f"Compute derivative of head sum for element {cell}")
    old_p = p[cell-1]
    p[cell-1] += old_p * pertub
    prop = parametrization.convert_p_to_mat_properties(p)
    solver.create_cell_indexed_dataset(prop, propname, propname+'.h5')
    solver.run()
    for j,var in enumerate(objective.__get_variables_needed__()):
      solver.get_output_variable(var,out=inputs[j])
    cur_obj = objective.evaluate(prop)
    deriv[i] = (cur_obj-ref_obj) / (old_p * pertub)
    p[cell-1] = old_p
  return deriv
