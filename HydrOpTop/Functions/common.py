import numpy as np

def __cumsum_from_connection_to_array__(array_out, sum_at, values, sorted_index=None):
  """
  Sum the values array in array out at the index defined by sum_at
  This is equivalent to array_out[sum_at] += values where sum_at can have redudant indices
  sum_at must be a zeros based array. 
  Negative index are ignored, i.e. do not sum the value associated with the the sum at
  index of -1
  """
  values_ = values[sum_at >= 0]
  sum_at_ = sum_at[sum_at >= 0] #remove -1
  if len(sum_at_) == 0: return #do nothing
  if sorted_index is None:
    sorted_index = np.argsort(sum_at_)
  #bincount sum_at+1 so there is no 0 and the output array[0] = 0
  tosum = np.bincount(sum_at_+1, minlength=len(array_out))
  n_to_sum = np.cumsum(tosum)[:-1]
  n_to_sum[n_to_sum == len(sum_at_)] = len(sum_at_) - 1 #post treatment for end missing cell ids
  where_to_add = np.where(tosum[1:] != 0)[0]
  if len(array_out.shape) == 1:
    array_out[where_to_add] += \
             np.add.reduceat(values_[sorted_index], n_to_sum)[where_to_add]
  else:
    array_out[where_to_add,:] += \
             np.add.reduceat(values_[sorted_index,:], n_to_sum)[where_to_add]
    
  return 
