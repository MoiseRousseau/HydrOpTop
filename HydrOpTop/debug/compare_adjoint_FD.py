import numpy as np 


def compare_adjoint_with_FD(craft_objet, p, cell_ids_to_check, pertub=1e-3):
  #compute gradient using adjoint
  print("Compute gradient using adjoint method")
  grad_adjoint = np.zeros(len(p), dtype="f8")
  objective = craft_objet.nlopt_function_to_optimize(p, grad_adjoint)
  
  #compute finite difference for cell_ids_to_check
  grad_FD = np.zeros(len(cell_ids_to_check), dtype='f8')
  cell_index = [np.where(craft_objet.p_ids == x)[0][0]
                           for x in cell_ids_to_check]
  for i,cell_id in enumerate(cell_ids_to_check):
    print("Compute gradient using finite difference for cell ", cell_id)
    old_p = p[cell_index[i]]
    p[cell_index[i]] = old_p * (1+pertub)
    new_objective = craft_objet.evaluate_objective(p)
    grad_FD[i] = (new_objective - objective) / (old_p*pertub)
    p[cell_index[i]] = old_p
  
  #outut result to user
  print("  Cell id\tGradient Adjoint\tGradient FD\tDifference")
  for i,cell_id in enumerate(cell_ids_to_check):
    index = cell_index[i]
    diff = 1 - np.abs(grad_FD[i]/grad_adjoint[index])
    print(f"  {cell_id}\t\t{grad_adjoint[index]:.6E}\t\t{grad_FD[i]:.6E}\t{diff:.3%}")
  
  return
