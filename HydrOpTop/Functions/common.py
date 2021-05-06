import numpy as np

def __cumsum_from_connection_to_array__(array_out, sum_at, values):
  not_summed = np.ones(len(values),dtype='bool')
  if len(sum_at) != len(values): 
    print("sum_at and values argument must have the same length")
    raise ValueError
  while True in not_summed:
    unique_sum_at, indexes = np.unique(sum_at[not_summed], return_index=True)
    array_out[unique_sum_at] += values[not_summed][indexes]
    i = np.argwhere(not_summed)
    not_summed[i[indexes]] = False
  return
  
if __name__ == "__main__":
  val = np.array([10,20,30,40,50,60])
  sum_at = np.array([1,2,0,2,2,1])
  out = np.zeros(3)
  __cumsum_from_connection_to_array__(out, sum_at, val)
  print(out)
