import numpy as np 


def compare_dfunction_dp_with_FD(obj, p, pertub=1e-6, accept=1e-3):
  """
  Test if the derivative computed by the function d_objective_dP() in a Function
  instance is correct versus finite difference
  Note: the function input must be set
  """
  #compute function derivative
  obj.d_objective_dp_partial(p)
  deriv = obj.dobj_dp_partial
  if isinstance(deriv, float): deriv = np.zeros(len(p),dtype='f8') + deriv
  
  ref_obj = obj.evaluate(p)
  deriv_fd = np.zeros(len(p),dtype='f8')
  #compute finite difference
  for i in range(len(p)):
    old_p = p[i]
    d_p = old_p * pertub
    p[i] = old_p + d_p
    deriv_fd[i] = (obj.evaluate(p)-ref_obj) / d_p
    p[i] = old_p
  
  #print results
  mask = np.where(deriv != 0)
  diff = abs(deriv - deriv_fd)
  diff[mask] = abs(1-deriv_fd[mask]/deriv[mask])
  index_max = np.argmax(diff)
  print(f"Max difference: {diff[index_max]} at id {index_max+1}")
  if diff[index_max] < accept: 
    print("OK")
    return 0
  else: 
    print("X\n")
    print("Cell id\tBuilt-in\tFinite Diff\tError")
    for i in range(len(deriv_fd)):
      print(f"{i+1}\t{deriv[i]:.6e}\t{deriv_fd[i]:.6e}\t{diff[i]:.6e}")
    return 1
    
