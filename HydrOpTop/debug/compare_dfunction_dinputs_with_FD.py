import numpy as np 
#import ..Materials as Materials


def compare_dfunction_dinputs_with_FD(obj, p, cell_to_check=None, pertub=1e-6, accept=1e-3,
                                      detailed_info = False):
  """
  Test if the derivative computed by the function d_objective_dP() in a Function
  instance is correct versus finite difference
  Note: the function inputs must be set
  """
  print("\nDEBUG: compute function derivative wrt inputs vs finite difference")
  if cell_to_check is None: cell_to_check = np.arange(1,len(p)+1)
  else: cell_to_check = np.array(cell_to_check)
  #compute function derivative
  obj.d_objective_d_mat_props(p)
  derivs = obj.dobj_dmat_props
  ref_obj = obj.evaluate(p)
  
  #set function inputs
  variables_name = obj.__get_PFLOTRAN_output_variable_needed__()
  variables = obj.get_inputs()
  
  #compute finite difference
  deriv_fd = np.zeros(len(cell_to_check),dtype='f8')
  err = 0
  for j,name in enumerate(variables_name):
    if name not in ["PERMEABILITY"]: 
      print(f"Skip variable \"{name}\" (not parametrized in Materials module)")
      continue
    variable = variables[j]
    for i,cell in enumerate(cell_to_check):
      old_variable = variable[cell-1] #in PFLOTRAN id!!!
      d_variable = old_variable * pertub
      variable[cell-1] = old_variable + d_variable
      deriv_fd[i] = (obj.evaluate(p)-ref_obj) / d_variable
      variable[cell-1] = old_variable
      
    #print results
    if not isinstance(derivs[j], np.ndarray):
      derivs[j] = np.zeros(len(cell_to_check),dtype='f8')
    else:
      derivs[j] = derivs[j][cell_to_check-1]
    mask = np.where(derivs[j] != 0) 
    diff = abs(derivs[j] - deriv_fd)
    diff[mask] = abs(1-deriv_fd[mask]/derivs[j][mask])
    mask = np.where(deriv_fd == 0.)
    diff[mask] = abs(derivs[j][mask] - deriv_fd[mask])
    index_max = np.argmax(diff)
    print(f"Max difference for variable \"{name}\": {diff[index_max]} at id {cell_to_check[index_max]}")
    if diff[index_max] < accept: 
      print("OK")
    else: 
      print("X\n")
      err = 1
      
    if err or detailed_info:
      print("Cell id\tBuilt-in\tFinite Diff\tError")
      for i,cell in enumerate(cell_to_check):
        print(f"{cell}\t{derivs[j][i]:.6e}\t{deriv_fd[i]:.6e}\t{diff[i]:.6e}")
  
  return err
    
