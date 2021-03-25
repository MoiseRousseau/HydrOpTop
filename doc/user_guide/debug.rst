.. _debug:

Debugging functions
===================

HydrOpTop comes with some debugging functions to compare the gradient calculated
using the adjoint equation with those calculated using finite difference.

``compare_dfunction_dp_with_FD(obj, p, cell_to_check, pertub, accept, detailed_info)``

``compare_adjoint_with_FD(craft_objet, p, cell_ids_to_check, pertub, accept)``

