import numpy as np
from .Base_Function_class import Base_Function

class p_Weighted_Sum(Base_Function):
    r"""
    Integrate the given field F ponderated value of the primary optimization variable `p` over the domain :math:`D` (i.e., compute the ratio of volume of material designed by `p=1`):

    .. math::
             
            f = \sum_{i \in D} p_i^n F_i

    :param field: The constant field F to sum as array of value
    :type field: iterable
    :param ids_to_sum: a list of cell ids on which to compute the volume percentage. If not provided, sum on all parametrized cells
    :type ids_to_sum: iterable
    :param index_by_p0: if set to ``True``, switch the material and rather calculate the field according to the fraction of the material designed by `p=0`. In this case, :math:`p_i` is remplaced by :math:`p'_i = 1-p_i`.
    :param penalization: The penalization argument `n`. Higher value favor the high field zone.
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

        # ask the crafter the density parameter p at the same place than where the field is defined
        self.indexes = field_ids

        self.variables_needed = []
        return
    
    
    ### COST FUNCTION ###
    def evaluate(self,p):
        """
        Evaluate the cost function
        Return a scalar of dimension [L**3]
        """
        if self.vp0: p_ = 1-p
        else: p_ = p
        cf = np.sum(self.field*np.power(p_,self.n))
        return cf

    
    def d_objective_dp_partial(self,p): 
    	# TODO: penalization
        res = np.zeros(len(p),dtype='f8')
        if self.vp0: 
            factor = -1.
        else:
            factor = 1.
        res = factor * self.field * self.n * p**(self.n-1)
        return res


    @classmethod
    def sample_instance(cls):
        N = 10
        f = np.random.random(N)
        cell_ids = np.arange(N)
        res1 = cls(field=f, field_ids=cell_ids, index_by_p0=False, penalization=1.)
        res2 = cls(field=f, field_ids=cell_ids, index_by_p0=False, penalization=2.)
        res3 = cls(field=f, field_ids=cell_ids, index_by_p0=True, penalization=1.)
        return [res1,res2,res3]
