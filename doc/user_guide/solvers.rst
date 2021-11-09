.. _pflotran:

Solver class
============

A Input/Output shield class to interact with a variety of solver.
Used to specify the input dataset, launching the simulation and
reading the output variables.
Currently, only two solver are supported.

PFLOTRAN
--------

A class to interact with the reactive transport flow simulator PFLOTRAN.
Constructor is:

``sim = PFLOTRAN(pflotranin, mesh_info=None)``

with ``pflotranin`` the name of the PFLOTRAN input file (mandatory), and
``mesh_info`` the path to a PFLOTRAN output file containing simulation
independant PFLOTRAN output variable such as the mesh informations (face area or
cell volume for example).
Providing mesh information can help reduce the size of PFLOTRAN output file
at every iteration, therefore saving time and increase the life of your SSD!

The below PFLOTRAN specific methods are available:

``sim.set_parallel_calling_command(n,command)``: Set the number of process 
:math:`n` to run and the command to call (default is `mpiexec.mpich`)

``sim.get_region_ids(name)``: Return a numpy 1D array containing the ID of the cell
belonging to the region `name` as in the PFLOTRAN input file.

``create_cell_indexed_dataset(X_dataset, dataset_name, h5_file_name="",
X_ids=None, resize_to=True)``:
Create a HDF5 cell indexed dataset in the file named ``h5_file_name`` with 
the data given in the numpy array ``X_dataset`` and writed in the group
``dataset_name``. If the dataset length does not match the grid size, 
the dataset can be extended to the grid size with the ``X_dataset`` assigned
to the cell ids ``X_ids`` and -999 elsewhere.



Two-Dimensional Linear Elasticity
---------------------------------

Located in source. Must be compiled before use.
Call by the command ``MinimalFEM``.
Input need a 2D mesh (``.mesh`` file), material property file (``.matprops``) and a boundary condition (``.bcs``).



