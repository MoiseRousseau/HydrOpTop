.. _dev_notes:

Implement your function
-----------------------

You can create your own constrains and pass it directly to the optimizer of 
your choice. However, if you want to use it in the `Crafter` class, each new 
constrain class are required to have several methods in HydrOpTop. They are:

*   `__need_p_cell_ids__(self)` which should return `True` if the correspondance 
    between `p` and the PFLOTRAN cell ids is needed, `False` in the opposite case.

*   `__get_PFLOTRAN_output_variable_needed__(self)` which return a list of the
    PFLOTRAN output variables needed for your new class operation. Variable should
    be orthographied as in OUTPUT card.

*   `set_inputs(self, inputs)` TODO

*   `set_p_cell_ids(self, cell_ids)`

*   `set_filter(self, filter)`

The following method is required by the `nlopt` library:

*   `evaluate(self, p, grad)` where `p` and `grad` are the argument passed by 
    `nlopt` optimizer. `p` is the material density parameter in the region to 
    optimize and `grad` the gradient of the constrain w.r.t. `p`. This function 
    should return the constrain value and the its derivative w.r.t. the density
    parameter `p`.

A template is provided in `HydrOpTop/Constrains` folder.


Developper's notes
------------------

To create your own function, you could copy some of implemented function in HydrOpTop. Below is some tip and decription of the variable used.

*   ``self.p_ids`` is the correspondance between the parametrized cell and the cell id in PFLOTRAN simulation.
    It is passed at the function by the Crafter using ``set_p_to_cell_ids()`` method.
    For example, ``self.p_ids[1]=22`` mean the second parametrized cell in HydrOpTop is the cell id 22 in PFLOTRAN

*   ``self.ids_p`` is the previous variable, i.e. the correspondance between the cell id in PFLOTRAN and the parametrized cell in HydrOpTop. 
    For example, ``self.ids_p[55]=8`` means the cell id 56 in PFLOTRAN is parametrized by the cell
