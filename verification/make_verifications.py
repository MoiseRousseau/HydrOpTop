
import subprocess

def make_verifications():
  tests = {
    "Functions derivative wrt p" : "test_functions/dfunction_dp.py",
    "Functions derivative wrt pressure" : "test_functions/dfunction_dpressure.py",
    "Functions derivative wrt inputs" : "test_functions/dfunction_dinputs.py",
    "Adjoint Total Derivative Standalone" : "adjoint_derivative/compare_adjoint_derivative.py",
    "Adjoint Total Derivative Subset with Debug" : 
              "adjoint_derivative/subset/compare_adjoint_derivative_subset.py",
    # TODO "Adjoint Total Derivative with Filter" : "",
    # TODO "Sum_Flux function comparison with PFLOTRAN" : "",
    "Run Optimization Implicit Grid" : "pflotran_grid/make_optimization_imp.py",
    "Run Optimization Explicit Grid" : "pflotran_grid/make_optimization_exp.py",
    "Density Filter" : "test_filter/test_density_filter.py"
  }
  
  success = 0
  error = 0
  print("\nHydrOpTop verification tests\n")
  log = open("verifications.log",'w')
  log.close()
  for name,src_file in tests.items():
    log = open("verifications.log",'a')
    log.write("\n=============================\n")
    log.write(f"\nTest {name}\n")
    log.close()
    print(f"{name}: ",end="",flush=True)
    folder = '/'.join(src_file.split('/')[:-1])
    script = src_file.split('/')[-1]
    cmd = ["python3",script]
    ret = subprocess.call(cmd, stdout=open("verifications.log",'a'), cwd=folder)
    log = open("verifications.log",'a')
    if ret == 0: 
      print("Success")
      log.write("\nSuccess\n")
      success += 1
    else: 
      print("ERROR!")
      log.write("\nTest Error! See above\n")
      error += 1
    log.close()
  
  print(f"\nTests conducted: {len(tests)}")
  print(f"Passed: {success}")
  print(f"Failed: {error}\n")
  log = open("verifications.log",'a')
  log.write("\n=============================\n")
  log.write(f"\nTests conducted: {len(tests)}\n")
  log.write(f"Passed: {success}\n")
  log.write(f"Failed: {error}")
  log.close()
  return


if __name__ == "__main__":
  make_verifications()
