#
# This script test the derivative of the different objective function
# relative to the pressure P and compare it to finite difference
# calculation
#

from common_compare import common_compare
from HydrOpTop.debug import compare_dfunction_dpressure_with_FD


if __name__ == "__main__":
  err = common_compare(compare_dfunction_dpressure_with_FD, pertub=0.14e-8, accept=5e-2)
  if err: exit(1)
  else: exit(0)

  
