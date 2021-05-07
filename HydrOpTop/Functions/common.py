import numpy as np
from scipy.sparse import coo_matrix

def __cumsum_from_connection_to_array__(array_out, sum_at, values):
  sum_at_ = sum_at[sum_at >= 0]
  values_ = values[sum_at >= 0]
  indexes = np.arange(0,len(sum_at_))
  M = coo_matrix( (values_, (sum_at_, indexes)), shape=(len(array_out),len(sum_at_)) )
  M = M.tocsr()
  array_out[:,np.newaxis] += M.sum(axis=1)
  return

def smooth_max_0_function(x, k=1e6):
  cutoff = 100 / k
  res = x.copy()
  res[x < cutoff] = 1/k*np.log(1+np.exp(k*x[x < cutoff]))
  return res

def d_smooth_max_0_function(x, k=1e6):
  cutoff = 100 / k
  res = np.ones(len(x), dtype='f8')
  res[x < cutoff] = 0.
  res[abs(x) < cutoff] = 1/(1+np.exp(-k*x[abs(x) < cutoff]))
  return res

def smooth_abs_function(x, k=1e6):
  return x*np.tanh(k*x)

def d_smooth_abs_function(x, k=1e6):
  tanh = np.tanh(k*x)
  return tanh + x*k*(1-tanh**2)
