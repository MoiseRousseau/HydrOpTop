import sys
import os
path = os.getcwd() + '/../../'
sys.path.append(path)

import numpy as np
from HydrOpTop.Functions import common

class Test_Common_Function:
  
  def test_cummulative_sum(self):
    val = np.array([10,20,30,40,50,60],dtype='i8')
    sum_at = np.array([1,2,0,2,2,1])
    out = np.zeros(3)
    common.__cumsum_from_connection_to_array__(out, sum_at, val)
    assert out[0] == 30
    assert out[1] == 70
    assert out[2] == 110
    
  def test_cummulative_sum(self):
    val = np.array([10,20,30,40,50,60,70],dtype='i8')
    sum_at = np.array([1,2,0,2,-1,1,2])
    out = np.zeros(3)
    common.__cumsum_from_connection_to_array__(out, sum_at, val)
    assert out[0] == 30
    assert out[1] == 70
    assert out[2] == 130
  
  def test_d_smooth_max_0_function(self):
    x = np.logspace(-5,5,11)
    x = np.append(-x,x)
    max_0_x = common.smooth_max_0_function(x)
    pertub = 1e-6
    x_pertub = (1+pertub)*x
    max_0_x_dx = common.smooth_max_0_function(x_pertub)
    deriv_fd = (max_0_x_dx-max_0_x) / (x_pertub-x)
    deriv = common.d_smooth_max_0_function(x)
    print(deriv, deriv_fd)
  
  def test_d_smooth_abs_function(self):
    x = np.logspace(-5,5,11)
    x = np.append(-x,x)
    abs_x = common.smooth_abs_function(x)
    pertub = 1e-6
    x_pertub = (1+pertub)*x
    abs_x_dx = common.smooth_abs_function(x_pertub)
    deriv_fd = (abs_x_dx-abs_x) / (x_pertub-x)
    deriv = common.d_smooth_abs_function(x)
    print(deriv, deriv_fd)
    
    
  
  
