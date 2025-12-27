import numpy as np
from scipy.sparse import diags_array, coo_array
from .Base_Filter_class import Base_Filter

class Filter_Sequence(Base_Filter):
    """
    Create a mapping between HydrOpTop density parameter and simulation density parameter through a list of filter.
    Handle a list of filters applied consecutively with helper routine like derivation
    All indexes are in solver indexing.
    
    :param filters: list of all filters
    :param parametrized_cell: list of all parametrized cells
    """
    def __init__(self, filters):
        super(Filter_Sequence, self).__init__()
        self.filters = filters
        self.input_dim = 0
        self.output_dim = 0
        self.input_ids = np.array([])
        self.output_ids = np.array([])
        
        input_ids = set()
        output_ids = set()
        for f in self.filters:
            # if f.input_ids is None:
            #     input_cell = set(self.solver.get_region_ids("__all__"))
            #     break
            # else:
            input_ids = input_ids.union([x for x in f.input_ids if x >= 0])
            # if f.output_ids is None:
            #     output_cell = set(self.solver.get_region_ids("__all__"))
            #     break
            # else:
            output_ids = output_ids.union(f.output_ids)
        # correct for not filtered cell
        self.input_ids = np.fromiter((x for x in input_ids), dtype='i4')#.union(not_filtered_cell))
        self.output_ids = np.fromiter((x for x in output_ids), dtype='i4')#.union(not_filtered_cell))
        self.input_dim = len(self.input_ids)
        self.output_dim = len(self.output_ids)
        # create reverse mapping
        self.sim_to_p_ids = np.ma.masked_all(self.output_ids.max() + 1, dtype="i4")
        self.sim_to_p_ids[self.output_ids] = np.arange(self.output_dim, dtype="i4")
        #self.output_cell = np.array(self.output_cell)
    
    def filter(self, p):
        return self.get_filtered_density(p)
    
    def get_filtered_density(self, p):
        p_bar = np.zeros(self.output_dim)
        p_bar[self.sim_to_p_ids[self.input_ids]] = p
        if not self.filters:
            return p_bar
        for filter_ in self.filters: #apply filter consecutively
            # Filter does not need to be updated
            p_bar[self.sim_to_p_ids[filter_.output_ids]] = filter_.get_filtered_density(
                p_bar[self.sim_to_p_ids[filter_.input_ids]]
            )
        return p_bar
    
    def filter_derivative(self, p):
        """
        Derivative of the filter sequence according to input p
        """
        # initialize filter state 
        p_bar = np.zeros(self.output_dim)
        p_bar[self.sim_to_p_ids[self.input_ids]] = p
        N = self.output_dim
        Jf = diags_array(np.ones(self.output_dim),format="csr")
        if not self.filters:
            return p_bar, Jf
        # Compute d_p_bar / d_p, integrate through filter
        for i,f in enumerate(self.filters):
            # We are in simulation space here
            # and filter acts as a bijection, we take the column we want at the end
            Jp = f.get_filter_derivative(p_bar[self.sim_to_p_ids[f.input_ids]])
            global_row = self.sim_to_p_ids[f.output_ids][Jp.coords[0]]
            global_col = self.sim_to_p_ids[f.input_ids][Jp.coords[1]]
            # add 1 to indexes unchanged by the filter
            uc = np.nonzero(~np.isin(
                np.arange(0,N), self.sim_to_p_ids[f.output_ids]
            ))[0]
            global_row = np.concat([global_row,uc])
            global_col = np.concat([global_col,uc])
            data = np.concat([Jp.data,np.ones(len(uc))])
            J = coo_array(
                (data,(global_row,global_col)), shape=(N,N)
            )
            # Update global filter derivative
            Jf = J.dot(Jf)
            # Update filter state
            p_bar[self.sim_to_p_ids[f.output_ids]] = f.get_filtered_density(
                p_bar[self.sim_to_p_ids[f.input_ids]]
            )
        return p_bar, Jf[:,self.sim_to_p_ids[self.input_ids]]
    
    @classmethod
    def sample_instance(cls):
        from . import Density_Filter, Heaviside_Filter
        instances = []
        # Test with one filter (cell_ids = parametrized_cells)
        N = 100
        cell_ids = np.arange(N)
        filter_1 = Density_Filter(cell_ids, radius=0.1, distance_weighting_power=1.)
        filter_1.inputs = {
            "ELEMENT_CENTER":np.random.random((N,2)),
            "VOLUME":np.random.random(N)
        }
        instance = cls([filter_1])
        instances.append(instance)
        # Test with 2 filters
        filter_2 = Heaviside_Filter(cell_ids)
        instance = cls([filter_1, filter_2])
        instances.append(instance)
        return instances