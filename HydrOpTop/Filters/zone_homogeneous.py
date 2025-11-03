from .Base_Filter_class import Base_Filter
import numpy as np
import scipy.sparse as sp

class Zone_Homogeneous(Base_Filter):
    r"""
    Assign the same density parameter value in all the given cell_ids, equivalent
    defining a zonal density parameter.
    In practice, tied all the cell_ids to the first cell_id given
    """
    def __init__(self, cell_ids_zone):
        super(Zone_Homogeneous, self).__init__()
        self.cell_ids = np.asarray(cell_ids_zone)
        self.input_ids = np.array([self.cell_ids[0]]) # does not depend on other variable on the mesh
        self.output_ids = np.asarray(cell_ids_zone)
        return

    def get_filtered_density(self, p):
        """
        Parameter
          p: the density parameter corresponding to the cell_ids given
        """
        return np.ones(len(self.output_ids))*p[0]

    def get_filter_derivative(self, p):
        """
        Return a n by n matrix with partial derivative of filter output relative to input
        We are in simulation numbering.
        p[i] represent the density parameter at cell i
        So dp_bar / dp at i 
        """
        n = len(self.cell_ids)
        col = np.zeros(n) #we tied it the first cell, so col=0
        row = np.arange(0,n)
        J = sp.coo_array(
            ( np.ones(n) , (row,col) ),
            shape=(n,1),
            dtype=np.double,
        )
        return J


    @classmethod
    def sample_instance(cls):
        insts = []
        N = 10
        cell_ids = np.arange(N)
        # create test
        instance = cls(cell_ids)
        instance.input_indexes = [0]
        instance.output_indexes = np.arange(len(cell_ids))
        insts.append(instance)
        return insts
