.. _installation:


Installation
============

HydrOpTop Python library contains only the necessary for solving inverse problem, and the forward problem solver (i.e. PFLOTRAN must be installed in a modified version). 


HydrOpTop
---------

HydrOpTop could be installed using Python ``pip`` command:

``pip install HydrOpTop``

Developpement version could be installed clone the HydrOpTop GitHub `repository <https://github.com/MoiseRousseau/HydrOpTop>`_.


PFLOTRAN Installation
---------------------

1. Follow instructions on how to install PFLOTRAN in the official `documentation <https://www.pflotran.org/documentation/user_guide/how_to/installation/linux.html#linux-install>`_ and up to the step 4 included.

2. In the terminal, copy-paste the following lines which download PFLOTRAN modifications, apply them and compile the executable:

.. code-block:: bash

  cd pflotran
  git checkout maint/v5.0
  wget https://raw.githubusercontent.com/MoiseRousseau/HydrOpTop/master/code_patches/PFLOTRAN-5.0.0-opt.patch
  git apply -3 PFLOTRAN-5.0.0-opt.patch
  cd src/pflotran
  make pflotran -j8

3. Add path to PFLOTRAN executable in your path:

.. code-block:: bash

  echo "alias pflotran=$(pwd)/pflotran" >> ~/.bashrc
