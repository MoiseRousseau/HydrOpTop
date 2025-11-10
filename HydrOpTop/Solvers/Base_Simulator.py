import h5py
import numpy as np

class Base_Simulator:
    def __init__(self, dry_run=False):
        self.name = "Base Simulator"
        self.dry_run = dry_run
        self.problem_size = 0
        self.cell_id_start_at = 0
        self.var_loc = "cells"
        return


    def get_region_ids(self, name):
        return np.arange(self.problem_size)

    def get_grid_size(self):
        return self.problem_size

    def get_var_location(self, var):
        return self.var_loc

    def get_mesh(self):
        vert = np.random.rand(self.problem_size,2)
        cells = []
        indexes = []
        return vert, cells, indexes

    def create_cell_indexed_dataset(self, X_dataset, dataset_name,
                                  X_ids=None, resize_to=True):
        r"""
        Create a HDF5 cell indexed dataset.
          
        :param X_dataset: The cell dataset to write
        :type X_dataset: iterable
        :param dataset_name: Name of the dataset (used to infer output file name)
        :type dataset_name: str
        :param X_ids: The cell ids matching the dataset value in X_dataset.
            (i.e. if X_ids = [5, 3] and X_dataset = [1e7, 1e8], therefore, cell id 5
            will have a X=1e7 and cell id 3 X=1e8). By default, assumed in natural order
        :type X_ids: iterable
        :param resize_to: boolean for resizing the given dataset to total number of cell and populate with NaN (default = True)
        :type resize_to: bool
        """
        #first cell is at i = 0
        h5_file_name = dataset_name.lower() + '.h5'
        out = h5py.File(h5_file_name, 'w')
        if X_ids is None: resize_to=False
        if resize_to and self.n_cells != len(X_dataset):
          if X_ids is None:
            print("Error: user must provide the cell ids corresponding to the dataset since the length of the dataset length does not match the number of cell in the grid")
            exit(1)
          X_new = np.zeros(self.n_cells, dtype='f8')
          X_new[X_ids.astype('i8')-1] = X_dataset
          X_dataset = X_new
          X_ids = None
        out.create_dataset(dataset_name, data=X_dataset)
        if X_ids is None:
          out.create_dataset("Cell Ids", data=np.arange(1,len(X_dataset)+1))
        else:
          out.create_dataset("Cell Ids", data=X_ids)
        out.close()
        return
