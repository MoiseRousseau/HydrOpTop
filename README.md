# HydrOpTop

HydrOpTop is a Python library which aims to provide a modular, flexible and solver-independant approach for topology optimization (TO) using the density-based approach.
It aims to be the Swiss army knife and a standard exchange place for state-of-the-art tools and engineering discpline involved in TO.

Solvers are interfaced using a I/O shield, which thus allow to define cost function, constraints and filters in a reusable manner for different softwares and codes.
Also, objective functions and constraints are implemented under the same class and without distinction, which means they can be interchanged effortlessly.
Base classes for cost functions/constraints, material parametrizations and filters are also provided so user may define they own TO features with a minimal amount of code.


## Getting started

### Documentation

The present README file contains some basic information about what HydrOpTop is, how to install it and to run some examples to see what HydrOpTop is capable of (see below).
For a deeper description of all its capacities, a larger documentation is located [here]() which the reader/user is refered.

Note this project is young, so documentation are still in development.


### Installation

HydrOpTop library can be easily installed using Python ``pip`` command as:

```
pip install HydrOpTop
```

Installation includes all the different HydrOpTop modules (material parametrization, functions, filters) including the solver I/O shields (see [here](https://TODO) for the list of interfaced solvers).
However, solvers are in general be included, this means users need to install it manually.
Solver installation instructions for HydrOpTop can be found in the documentation [here]().

[Paraview](https://www.paraview.org/) software is also recommanded to visualize HydrOpTop results.

Again, please note that as the project is young, so features, commands or capabilities may change fast. 
Stay in tune!


## Examples

The ``examples`` folder contains some classic topology optimization problem to illustrate what HydrOpTop is capable of.

### Linear elasticity examples

* **Cantilever_simple**: The classical TO benchmark using a homemade 2D linear elasticity FEM solver. Consists in minimizing the mechanical compliance of a cantilever under a maximum volume constraint.
* **Cantilever_discrete**: The same previous classical benchmark but using a Heavyside projection to create a discrete final design.
* **Cantilever_min_volume**: A variation of the classical benchmark consisting in minimizing the weight (i.e. the volume) of the cantilever under a maximum mechanical compliance constraint.

### Hydrogeology examples (with PFLOTRAN)

* **Pond_drainage**: TODO
* **Reactive_barrier_max_flow**: TODO
* **Pit_min_gradient**: TODO



## Contributing

The library aims to be an exchange place for tools and between various engineering disciplines. 
Therefore, your contributions are warmly welcomed and will be acknowledged.
Guide for developpers and contribution rules are summarized in the documentation [here]().
However, if you are a regular user, you can still help by reporting bugs and issues (see the GitHub issue tab above) or by starring this project if you like it. Thanks for your implication!


## Authors

* **Moïse Rousseau** - *Initial work*: initially started as a part of my PhD thesis at the [Research Institute of Mines and Environment](https://irme.ca/en/), continued latter as a personnal project.
