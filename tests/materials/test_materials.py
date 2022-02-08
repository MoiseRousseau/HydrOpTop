#
# Verification of the sensitivity of the filtered density and derivative calculation
#

import sys
import os
path = os.getcwd() + '/../../'
sys.path.append(path)

import numpy as np

import HydrOpTop.Materials as M
from scipy.misc import derivative
import pytest


class Test_Materials:

  test_mat = [M.SIMP, M.Log_SIMP, M.RAMP]
  
  @pytest.mark.parametrize("mat", test_mat)
  def test_derivative(self, mat):
    param = mat("all","TEST",[0.1,1])
    pts = [0.12, 0.5, 0.7, 1]
    ana = param.d_mat_properties(np.array(pts,dtype='f8'))
    num = np.array([derivative(param.convert_p_to_mat_properties, np.array([x]), dx=1e-6)[0] for x in pts])
    assert np.allclose(num, ana)
    
