.. _examples_flow_reactive_barrier:

Maximize flow in a reactive barrier
===================================

Problem description
-------------------

The problem consists of an excavation to be backfilled with hazardous waste.
Waste will release a potentially harmful chemical into the environment, that we 
want to treat using an already constructed reactive barrier nearby.
The objective is to place a highly conductive material in the excavation and within
the waste such as the groundwater flow is directed toward the barrier.

Figure of the problem

Optimization problem
--------------------

Try to direct the flow toward the barrier is considered equivalent to maximize
the groundwater flowrate through it. 
Therefore, the `Sum_Flux` objective function is used. 
A maximum volume percentage of 15% of highly conductive material is also applied
such as at least 85% of the excavation volume is reserved for the waste disposal.
The optimization problem can be write as follow:

The mat problem

Python code
-----------

Copy-paste of the code

Results
-------

Figure + comment
