#
# This script test the derivative of the different objective function
# relative to the material parameter p and compare it to finite difference
# calculation
#

from common_compare import common_compare
from HydrOpTop.debug import compare_dfunction_dp_with_FD


if __name__ == "__main__":
  err = common_compare(compare_dfunction_dp_with_FD, pertub=1e-6, accept=1e-3)
  if err: exit(1)
  else: exit(0)

  
