
def __add_inputs__(obj, sim):
  inputs = {var:None for var in obj.__get_variables_needed__()}
  sim.get_output_variables(inputs)
  obj.set_inputs(inputs)
  return
