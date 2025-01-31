import numpy as np
from .Base_Function_class import Base_Function

class p_Weighted_Sum(Base_Function):
    r"""
    Integrate the given field F ponderated value of the primary optimization variable `p` over the domain :math:`D` (i.e., compute the ratio of volume of material designed by `p=1`):

    .. math::
             
            f = \sum_{i \in D} p_i^n F_i

    :param field: The field F to sum as array of value
    :type field: iterable
    :param ids_to_sum: a list of cell ids on which to compute the volume percentage. If not provided, sum on all parametrized cells
    :type ids_to_sum: iterable
    :param index_by_p0: if set to ``True``, switch the material and rather calculate the field according to the fraction of the material designed by `p=0`. In this case, :math:`p_i` is remplaced by :math:`p'_i = 1-p_i`.
    :param penalization: The penalization argument `n`.
    """
    def __init__(self, 
        field,
        field_ids=None,
        index_by_p0=False,
        penalization=1.,
    ):     
        super(p_Weighted_Sum, self).__init__()
        self.field = field
        self.field_ids = field_ids
        self.n = penalization
        self.vp0 = index_by_p0 #boolean to compute the volume of the mat p=1 (False) p=0 (True)
        self.name = "p Weighted Sum"
        return
    
    
    def set_inputs(self, inputs):
        return
    
    def get_inputs(self):
        return []
        
    def set_p_to_cell_ids(self, cell_ids):
        self.p_ids = cell_ids
        return
    
    
    ### COST FUNCTION ###
    def evaluate(self,p):
        """
        Evaluate the cost function
        Return a scalar of dimension [L**3]
        """
        if self.vp0: p_ = 1-p
        else: p_ = p
        cond = np.isin(self.p_ids, self.field_ids)
        f_in = np.where(np.in1d(self.field_ids,self.p_ids))[0]
        p_in = np.where(np.in1d(self.p_ids,self.field_ids))[0]
        cf = np.sum(self.field[f_in]*np.power(p_[p_in],self.n))
        return cf

    
    def d_objective_dp_partial(self,p): 
    	# TODO: penalization
        res = np.zeros(len(p),dtype='f8')
        if self.vp0: 
            factor = -1.
        else:
            factor = 1.
        f_in = np.where(np.in1d(self.field_ids,self.p_ids))[0]
        p_in = np.where(np.in1d(self.p_ids,self.field_ids))[0]
        res[p_in] = factor * self.field[f_in]
        return res

