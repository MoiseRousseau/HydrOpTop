import numpy as np 


def compare_adjoint_with_FD(craft_objet, p, cell_ids_to_check, pertub=1e-3):
  #compute gradient using adjoint
  print("Compute gradient using adjoint method")
  grad_adjoint = np.zeros(len(p), dtype="f8")
  objective = craft_objet.nlopt_function_to_optimize(p, grad_adjoint)
  
  #compute finite difference for cell_ids_to_check
  grad_FD = np.zeros(len(cell_ids_to_check), dtype='f8')
  if craft_objet.p_ids is not None:
    cell_index = [-1 for x in cell_ids_to_check]
    for i,x in enumerate(cell_ids_to_check):
      index = np.where(craft_objet.p_ids == x)[0]
      if len(index) == 0:
        print(f"Cell id {x} is not a part of the optimized domain, pass")
      else:
        cell_index[i] = index[0]
  else: 
    cell_index = [x-1 for x in cell_ids_to_check]
  for i,cell_id in enumerate(cell_ids_to_check):
    if cell_index[i] == -1: continue #pass this cell
    print("Compute gradient using finite difference for cell ", cell_id)
    old_p = p[cell_index[i]]
    p[cell_index[i]] = old_p * (1+pertub)
    new_objective = craft_objet.nlopt_function_to_optimize(p, np.zeros(0))
    grad_FD[i] = (new_objective - objective) / (old_p*pertub)
    p[cell_index[i]] = old_p
  
  #outut result to user
  max_error = -1
  print("\n  Cell id\tGradient Adjoint\tGradient FD\tDifference")
  for i,cell_id in enumerate(cell_ids_to_check):
    index = cell_index[i]
    if index == -1: continue
    diff = 1 - np.abs(grad_FD[i]/grad_adjoint[index])
    if diff > max_error: max_error = diff
    print(f"  {cell_id}\t\t{grad_adjoint[index]:.6E}\t\t{grad_FD[i]:.6E}\t{diff:.3%}")
  
  if max_error > 0.01: return 1
  else: return 0
