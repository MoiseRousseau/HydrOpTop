import numpy as np
from scipy.sparse import coo_matrix

class Base_Filter:
    
    skip_test = False
  
    def __init__(self):
        self.p_ids = None
        self.name = "Base Filter"
        self.input_ids = None # The input cell_id (in simulation) to apply filter on
        self.output_ids = None # The output cell_id (in simulation)  that the filter compute values
        self.input_indexes = None # The input index to take p from
        self.output_indexes = None # The output index to write p
        self.adjoint = None
        self.variables_needed = []
        self.inputs = {}
  
    def set_inputs(self, inputs):
        self.inputs = inputs
        return
  
    def get_filtered_density(self, p):
        """
        TO DEFINE
        """
        return 1.

    def get_filter_derivative(self, p, eps=1e-6, drop_tol=1e-4):
        """
        Compute derivative of p_bar relative to p with centered finite difference.

        Return a matrix with n_rows = output_dim, n_cols = input_dim

        This naive version use a finite difference approach and a dense jacobian.

        :param p: Density parameter p
        :param eps: Absolute step for finite difference calculation

        """
        print("Compute filter derivative with finite difference method...")
        data = []
        rows = []
        cols = []

        for i in range(len(p)):
            x1 = p.copy()
            x2 = p.copy()

            x1[i] += eps
            x2[i] -= eps

            f1 = self.get_filtered_density(x1)
            f2 = self.get_filtered_density(x2)

            df = (f1 - f2) / (2 * eps)

            # Only keep entries above drop_tol
            nz_idx = np.abs(df) > drop_tol
            data.extend(df[nz_idx])
            rows.extend(np.where(nz_idx)[0])
            cols.extend([i]*np.sum(nz_idx))

        # Construct COO sparse matrix
        J = coo_matrix((data, (rows, cols)), shape=(len(self.output_ids), len(p)), dtype=np.double)
        return J

    def plot_filtered_density(self, ax=None, show=True):
        try:
            import matplotlib.pyplot as plt
        except:
            raise ImportError("Please install Matplotlib to plot filter curve")
        x = np.linspace(0,1,1000)
        y = self.get_filtered_density(x)
        if ax is None: fig,ax = plt.subplots()
        ax.plot(x,y,'b',label="Filtered parameter")
        ax.set_xlabel("Input Parameter")
        ax.set_ylabel("Filtered Parameter")
        ax.grid()
        if show: plt.show()
        return


    def __get_variables_needed__(self):
        return self.variables_needed
    def __get_name__(self):
        return self.name
  

