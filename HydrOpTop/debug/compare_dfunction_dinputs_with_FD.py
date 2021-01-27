import numpy as np 
#import ..Materials as Materials


def compare_dfunction_dinputs_with_FD(obj, p, cell_to_check=None, pertub=1e-6, accept=1e-3,
                                      detailed_info = False):
  """
  Test if the derivative computed by the function d_objective_dP() in a Function
  instance is correct versus finite difference
  Note: the function inputs must be set
  """
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
  deriv_fd = np.zeros(len(p),dtype='f8')
  p_ids = obj.p_ids
  err = 0
  for j,name in enumerate(variables_name):
    if name not in ["PERMEABILITY"]: 
      print(f"Skip variable \"{name}\" (not parametrized in Materials module)")
      continue
    variable = variables[j]
    for cell in cell_to_check:
      if cell not in p_ids: 
        print(f"Cell ids {cell} not parametrized, can't compute derivative at this cell")
        continue
      ii = np.where(p_ids == cell)[0][0]
      old_variable = variable[cell-1] #in PFLOTRAN id!!!
      d_variable = old_variable * pertub
      variable[cell-1] = old_variable + d_variable
      deriv_fd[ii] = (obj.evaluate(p)-ref_obj) / d_variable
      variable[cell-1] = old_variable
      
    #print results
    if not isinstance(derivs[j], np.ndarray):
      derivs[j] = np.zeros(len(p),dtype='f8')
    mask = np.where(derivs[j] != 0) 
    diff = abs(derivs[j] - deriv_fd)
    diff[mask] = abs(1-deriv_fd[mask]/derivs[j][mask])
    mask = np.where(deriv_fd == 0.)
    diff[mask] = abs(derivs[j][mask] - deriv_fd[mask])
    index_max = np.argmax(diff)
    print(f"Max difference for variable \"{name}\": {diff[index_max]} at id {p_ids[index_max]}")
    if diff[index_max] < accept: 
      print("OK")
    else: 
      print("X\n")
      err = 1
      
    if err or detailed_info:
      print("Cell id\tBuilt-in\tFinite Diff\tError")
      for i,cell in enumerate(cell_to_check):
        if cell not in p_ids: continue
        index = np.where(p_ids == cell)[0][0]
        print(f"{cell}\t{derivs[j][index]:.6e}\t{deriv_fd[index]:.6e}\t{diff[index]:.6e}")
  
  return err
    
