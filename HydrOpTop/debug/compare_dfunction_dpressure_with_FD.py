import numpy as np 


def compare_dfunction_dpressure_with_FD(obj, p, pertub=1e-6, accept=1e-3, detailed_info=False):
  """
  Test if the derivative computed by the function d_objective_dP() in a Function
  instance is correct versus finite difference
  Note: the function input must be set
  """
  #compute function derivative
  obj.d_objective_dP(p)
  deriv = obj.dobj_dP
  if deriv is None: deriv = np.zeros(len(p),dtype='f8')
  
  #get pressure from the function input
  string = "LIQUID_PRESSURE"
  try:
    index = obj.__get_PFLOTRAN_output_variable_needed__().index(string)
  except:
    #function does not depend on pressure
    index = -1
  
  #compute finite difference
  deriv_fd = np.zeros(len(deriv),dtype='f8')
  if index != -1:
    pressure = obj.get_inputs()[index]
    ref_obj = obj.evaluate(p)
    for i in range(len(pressure)):
      #note pressure is a pointer, so don't need to update the inputs
      old_pressure = pressure[i]
      d_pressure = old_pressure * pertub
      pressure[i] = old_pressure + d_pressure
      deriv_fd[i] = (obj.evaluate(p)-ref_obj) / d_pressure
      pressure[i] = old_pressure
  
  #print results
  mask = np.where(deriv != 0)
  diff = abs(deriv - deriv_fd)
  diff[mask] = abs(1-deriv_fd[mask]/deriv[mask])
  index_max = np.argmax(diff)
  print(f"Max difference: {diff[index_max]} at id {index_max+1}")
  ret_code = 0
  if diff[index_max] < accept: 
    print("OK")
  else: 
    print("X\n")
    ret_code = 1
  if ret_code or detailed_info:
    print("Cell id\tBuilt-in\tFinite Diff\tError")
    for i in range(len(deriv_fd)):
      print(f"{i+1}\t{deriv[i]:.6e}\t{deriv_fd[i]:.6e}\t{diff[i]:.6e}")
  
  return ret_code
    
