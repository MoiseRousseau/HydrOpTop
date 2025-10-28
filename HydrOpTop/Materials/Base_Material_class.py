import numpy as np
import matplotlib.pyplot as plt

class Base_Material:
    def __init__(self):
        self.indexes = None
        self.name = None
        self.cell_ids = None
        return
    
    def get_cell_ids_to_parametrize(self):
        return self.cell_ids
    
    def get_name(self):
        return self.name
    
    def convert_p_to_mat_properties(self, p, out=None):
        raise NotImplementedError()
    
    def d_mat_properties(self, p, out=None):
        # Do by finite difference
        raise NotImplementedError()
        return
    
    def convert_mat_properties_to_p(self, mat_prop_val):
        # Do by root finding
        raise NotImplementedError()
    
    def plot_parametrization(self):
        p = np.linspace(0.,1.,100)
        K = self.convert_p_to_mat_properties(p)
        dK = self.d_mat_properties(p)
        fig, ax = plt.subplots()
        ax.plot(p,K,'r', label="value")
        ax2 = ax.twinx()
        ax2.plot(p,dK,'b',label="derivative")
        ax.set_xlabel("(Filtered) Density parameter p")
        ax.set_xlim([0,1])
        ax.set_ylabel(f"{self.name}")
        ax2.set_ylabel(f"d {self.name} / dp")
        ax.grid()
        h1, l1 = ax.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        ax.legend(h1+h2, l1+l2)
        plt.tight_layout()
        plt.show()