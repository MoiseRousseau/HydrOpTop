.. _pflotran:

PFLOTRAN class
==============

`sim = PFLOTRAN(pflotranin)`

A class to interact with the software PFLOTRAN, both for the input data
and output variable.

The different method are implemented

`sim.set_parallel_calling_command(n,command)`: Set the number of process :math:`n` to run
and the command to call (default is `mpiexec.mpich`)

`sim.get_region_ids(name)`: Return a numpy 1D array containing the ID of the cell
belonging to the region `name` as in the PFLOTRAN input file.

`create_cell_indexed_dataset(data,data_name,out_file)`: 


