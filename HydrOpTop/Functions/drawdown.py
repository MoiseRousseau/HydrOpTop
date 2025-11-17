# This file is a part of the topology optimization program by Moise Rousseau

import numpy as np
from .Base_Function_class import Base_Function
from .function_utils import df_dX
from scipy.interpolate import LinearNDInterpolator

class Drawdown(Base_Function):
    r"""
    """
    def __init__(self, ref_head, cell_ids=None, XYZ_coordinates=None):
        super(Drawdown, self).__init__()
        
        self.ref_head = ref_head
        self.cell_ids = np.asarray(cell_ids) if cell_ids is not None else None
        self.XYZ = None
        if self.cell_ids is None:
            self.XYZ = np.asarray(XYZ_coordinates) if XYZ_coordinates is not None else None
        if self.cell_ids is None and self.XYZ is None:
            raise ValueError("Both cell_ids and XYZ coordinates parameter cannot be null at the same time. If you passed it by name, please use solver function to get either cell_ids or coordinate and retry.")
        if (self.cell_ids is None) == (self.XYZ is None):
            print("Both cell_ids and coordinates passed, rely only on cell_ids...")
            self.XYZ = None
        self.h_interpolator = None

        #required for problem crafting
        self.variables_needed = ["LIQUID_HEAD"]
        if self.XYZ is not None:
            # We need to construct an interpolator
            # Ask for coordinate of the head and the head
            self.variables_needed = [
                "MESH_VERTICE_XYZ",
                "LIQUID_HEAD_AT_VERTICE"
            ]
        if self.cell_ids is not None:
            self.indexes = self.cell_ids
        else:
            self.indexes = None #mean need all data from simulator
        self.name = "Drawdown"
        self.initialized = False
        return

    
    ### COST FUNCTION ###
    def __evaluate_cell_ids__(self,p):
        head = self.inputs["LIQUID_HEAD"]
        return np.sum(self.ref_head - head)
        
    def __evaluate_xyz__(self,p):
        self.h_interpolator.values[:,0] = self.inputs["LIQUID_HEAD_AT_VERTICE"]
        heads = self.h_interpolator(self.XYZ)
        r = heads - self.ref_head
        return np.sum(r)

    def evaluate(self, p):
        """
        Evaluate the cost function
        Return a scalar of dimension [L]
        """
        if not self.initialized: self.initialize()
        if self.cell_ids is not None:
            res = self.__evaluate_cell_ids__(p)
        else:
            res = self.__evaluate_xyz__(p)
        return res
    
    
    ### PARTIAL DERIVATIVES ###
    def __d_objective_dh_cell_ids__(self, p):
        """
        Given a variable, return the derivative of the cost function according to that variable.
        Use current simulation state
        """
        head = self.inputs["LIQUID_HEAD"]
        # Derivative according to the head
        return np.zeros_like(head) - 1.

    def __d_objective_dh_xyz__(self, p):
        """
        Given a variable, return the derivative of the cost function according to that variable.
        Use current simulation state
        """
        dobj = np.sum([
            df_dX(self.h_interpolator, xyz) for i,xyz in enumerate(self.XYZ)
        ], axis=0)
        return dobj


    def d_objective(self, var, p):
        """
        Given a variable, return the derivative of the cost function according to that variable.
        Use current simulation state
        """
        if not self.initialized: self.initialize()
        if self.cell_ids is not None and var == "LIQUID_HEAD":
            res = self.__d_objective_dh_cell_ids__(p)
        elif var == "LIQUID_HEAD_AT_VERTICE":
            res = self.__d_objective_dh_xyz__(p)
        else:
        # The function depends of no other variables
            res = np.zeros_like(p, dtype='f8')
        return res

    def initialize(self):
        self.initialized = True
        if self.XYZ is None:
            return
        xyz = self.inputs[f"MESH_VERTICE_XYZ"]
        self.h_interpolator = LinearNDInterpolator(xyz, self.inputs[f"LIQUID_HEAD_AT_VERTICE"])
        return

    @classmethod
    def sample_instance(cls):
        # sample cell_ids
        res1 = cls(ref_head=[11.2,43.2,56.4,29.4], cell_ids=[2,4,5,10])
        res1.set_inputs({"LIQUID_HEAD":np.random.rand(20)[res1.cell_ids]*100})
        # sample xyz
        from scipy.interpolate import LinearNDInterpolator
        res2 = cls(
            ref_head=[11.2,43.2,56.4,29.4],
            XYZ_coordinates=np.random.rand(4,3),
        )
        res2.set_inputs({
            "LIQUID_HEAD_AT_VERTICE":np.random.rand(20)*100,
            "MESH_VERTICE_XYZ": np.random.rand(20,3)*3-1
        })
        res2.deriv_var_to_skip = ["MESH_VERTICE_XYZ"]
        return [res1,res2]
