
def __add_inputs__(obj, sim):
  inputs = []
  for output in obj.__get_all_variables_needed__():
    inputs.append(sim.get_output_variable(output))
  obj.set_inputs(inputs)
  return
