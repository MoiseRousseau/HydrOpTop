import numpy as np 
#import ..Materials as Materials


def compare_dfunction_dinputs_with_FD(obj, p, cell_to_check=None, pertub=1e-6, 
                                      detailed_info = False):
  """
  Test if the derivative computed by the function d_objective_dP() in a Function
  instance is correct versus finite difference
  Note: the function input must be set
  """
  if cell_to_check is None: cell_to_check = np.arange(len(p))
  else: cell_to_check = np.array(cell_to_check)-1
  #compute function derivative
  obj.d_objective_d_mat_props(p)
  derivs = obj.dobj_dmat_props
  ref_obj = obj.evaluate(p)
  
  #set function inputs
  variables_name = obj.__get_PFLOTRAN_output_variable_needed__()
  variables = obj.get_inputs()
  
  #compute finite difference
  err = 0
  deriv_fd = np.zeros(len(p),dtype='f8')
  for j,name in enumerate(variables_name):
    if name not in ["PERMEABILITY"]: 
      print(f"Skip variable \"{name}\" (not parametrized in Materials module)")
      continue
    variable = variables[j]
    for i in cell_to_check:
      old_variable = variable[i]
      d_variable = old_variable * pertub
      variable[i] = old_variable + d_variable
      deriv_fd[i] = (obj.evaluate(p)-ref_obj) / d_variable
      variable[i] = old_variable
      
    
    #print results
    if not isinstance(derivs[j], np.ndarray):
      derivs[j] = np.zeros(len(p),dtype='f8')
    mask = np.where(derivs[j] != 0) 
    diff = abs(derivs[j] - deriv_fd)
    diff[mask] = abs(1-deriv_fd[mask]/derivs[j][mask])
    mask = np.where(deriv_fd == 0.)
    diff[mask] = abs(derivs[j][mask] - deriv_fd[mask])
    diff = diff[cell_to_check]
    index_max = np.argmax(diff)
    print(f"Max difference for variable \"{name}\": {diff[index_max]} at id {index_max+1}")
    if diff[index_max] < 1e-4 and not detailed_info: 
      print("OK")
    else: 
      print("X\n")
      print("Cell id\tBuilt-in\tFinite Diff\tError")
      for i,cell in enumerate(cell_to_check):
        print(f"{cell+1}\t{derivs[j][cell]:.6e}\t{deriv_fd[cell]:.6e}\t{diff[i]:.6e}")
      return 1
  
  return err
    
