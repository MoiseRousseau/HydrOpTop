from .Base_Filter_class import Base_Filter
from scipy.sparse import eye

class Unit_Filter(Base_Filter):
  """
  A dummy filter that do nothing but show the basic method for a filter
  """
  #skip_test = True
  
  def __init__(self, input_ids, output_ids=None):
    super(Unit_Filter, self).__init__()
    self.input_ids = input_ids
    if output_ids is None: output_ids = input_ids
    self.output_ids = output_ids
    return
  
  def get_filtered_density(self, p):
    return p.copy()
  
  def get_filter_derivative(self, p, out=None):
    return eye(len(p),format='coo')
  
  @classmethod
  def sample_instance(cls):
    insts = []
    cell_ids = [i for i in range(0,10)]
    # create test
    instance = cls(cell_ids)
    insts.append(instance)
    return insts
