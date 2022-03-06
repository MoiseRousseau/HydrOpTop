# MinimalFem for topology optimization

From [https://github.com/MoiseRousseau/MinimalFem-For-Topology-Optimization/](https://github.com/MoiseRousseau/MinimalFem-For-Topology-Optimization/)

Finite element modelling of 2D elastic problem in two dimensions on triangular grids, and considering heterogeneous distribution on material properties with Jacobian and sensitivity output (for topology optimization).
Derived from the original work at [https://github.com/podgorskiy/MinimalFem](https://github.com/podgorskiy/MinimalFem) and [https://podgorskiy.com/spblog/304/writing-a-fem-solver-in-less-the-180-lines-of-code](https://podgorskiy.com/spblog/304/writing-a-fem-solver-in-less-the-180-lines-of-code).


## Getting started

1. Download or clone this repository
2. Install the Eigen library (`sudo apt install libeigen3-dev` on Ubuntu)
3. Run in a terminal `cmake . && make`


## Use

The solver is called by the command: 
```
./MinimalFem <prefix>
```

It solves the displacements (`<prefix>.displacements` output) and the Von-Misses stresses  (`<prefix>.stress`) of a two-dimensional part according to the static linear elasticity theory and using finite element analysis.
It requires a 2D triangular mesh (`<prefix>.mesh` input), the position of the load and the boundary conditions (`<prefix>.bcs`) and the heterogeneous material property distribution (Poisson ratio and Young modulus in `<prefix>.matprops`).
It also stores the derivative of the linear system solved relative to the displacements  (`<prefix>_jacobian.mtx` file) and to the Yound modulus (`<prefix>_sensitivity.mtx` file) allowing solving adjoint problem and carrying topology optimization.


### Inputs

Mesh file (`<prefix>.mesh`) must be organized as follow:
```
n_v   #number of vertices
X1 Y1 #coordinate first vertice
...
Xi Yi #coordinate vertice i
... 
Xn_v Yn_v #coordinate last vertice (n_v)
n_t #number of triangles
u1 v1 w1 #the three vertices of triangle 1
...
uj vj wj #the three vertices of triangle j
...
un_t vn_t wn_t #the three vertices of triangle n_t
```

Loads and boundary conditions (`<prefix>.bcs`) as:
```
n_constraint #number of constraint on the node mouvement
v1 type1 #node number / type = 1: no mouvement in x, type = 2: no mouvement in y, type = 3 no mouvement in x,y
...
vn_constraint typen_constraint
n_loads
v1 dx1 dy1 #vertice number / load in x / load in y
...
vn dxn_loads dyn_loads
```

Material properties (`<prefix>.matprops`) as:
```
poisson_ratio #constant
ym1 #young modulus for triangle 1 (defined by u1 v1 w1 in the mesh file)
...
ymi #young modulus for triangle i
...
ymn_t #young modulus for triangle n_triangles
```

For the two first, also see the original blog [post](https://podgorskiy.com/spblog/304/writing-a-fem-solver-in-less-the-180-lines-of-code) section `input data`.



### Outputs

Displacements are stored in the following format:
```
dx1 dy1 #displacement at first vertice
...
dxi dyi #displacement at vertice i (line i)
...
dxn_v dyn_v #displacement at last vertice
```

Von-Misses stresses as:
```
s1 #stress at triangle 1 (u1 v1 w1)
...
si #stress at triangle i (line i)
..
sn_t #stress at triangle n_t
```

Jacobian (derivative of linear system relative to displacement) and sensitivity (derivative relative to Young modulus) are stored using Eigen `write_binary_sparse` function. 
Both are sparse matrix of size `(2*n_v,n_t)`.
