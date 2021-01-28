import numpy as np 


def compare_dfunction_dp_with_FD(obj, p, cell_to_check=None, pertub=1e-6, accept=1e-3,
                                 detailed_info = False):
  """
  Test if the derivative computed by the function d_objective_dP() in a Function
  instance is correct versus finite difference
  Note: the function input must be set
  """
  print("\nDEBUG: compute function derivative wrt material parameter vs finite difference")
  if cell_to_check is None: cell_to_check = np.arange(1,len(p)+1)
  else: cell_to_check = np.array(cell_to_check)
  #compute function derivative
  obj.d_objective_dp_partial(p)
  deriv = obj.dobj_dp_partial
  if isinstance(deriv, float): deriv = np.zeros(len(p),dtype='f8') + deriv
  
  ref_obj = obj.evaluate(p)
  deriv_fd = np.zeros(len(p),dtype='f8')
  #compute finite difference
  for cell in cell_to_check:
    if cell not in obj.p_ids: 
      print(f"Cell ids {cell} not parametrized, can't compute derivative at this cell")
      continue
    ii = np.where(obj.p_ids == cell)[0][0]
    old_p = p[ii]
    d_p = old_p * pertub
    p[ii] = old_p + d_p
    deriv_fd[ii] = (obj.evaluate(p)-ref_obj) / d_p
    p[ii] = old_p
  
  #print results
  mask = np.where(deriv != 0)
  diff = abs(deriv - deriv_fd)
  diff[mask] = abs(1-deriv_fd[mask]/deriv[mask])
  index_max = np.argmax(diff[np.isin(obj.p_ids,cell_to_check)])
  err = 0
  print(f"Max difference: {diff[index_max]} at id {obj.p_ids[index_max]}")
  if diff[index_max] < accept: 
    print("OK")
  else: 
    print("X\n")
    err = 1
  
  if err or detailed_info:
    print("Cell id\tBuilt-in\tFinite Diff\tError")
    for cell in cell_to_check:
      if cell not in obj.p_ids: continue
      index = np.where(obj.p_ids == cell)[0][0]
      print(f"{cell}\t{deriv[index]:.6e}\t{deriv_fd[index]:.6e}\t{diff[index]:.6e}")
  return err
    
