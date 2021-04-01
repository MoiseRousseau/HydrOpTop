
import subprocess

def make_verifications():
  tests = {
    "Functions derivative wrt p" : "test_functions_derivative/dfunction_dp.py",
    "Functions derivative wrt pressure" : "test_functions_derivative/dfunction_dpressure.py",
    "Functions derivative wrt inputs" : "test_functions_derivative/dfunction_dinputs.py",
    "Sum_Flux returned value" : "test_functions_value/sum_flux.py",
    "p_Weighted_Sum_Flux returned value": "test_functions_value/weighted_flux_vs_sum_flux.py",
    "Sum_Pz_Head Adjoint Derivative (Standalone)" : 
              "test_adjoint_derivative/sum_pz_head/compare_adjoint_derivative.py",
    "Sum_Pz_Head Adjoint Derivative on Subset" : 
              "test_adjoint_derivative/sum_pz_head/compare_adjoint_derivative_subset.py",
    "Sum_Flux Adjoint Derivative" : 
       "test_adjoint_derivative/sum_flux/compare_adjoint_derivative_sum_flux.py",
    "p_Weighted_Sum_Flux Adjoint Derivative" : 
       "test_adjoint_derivative/p_weighted_sum_flux/compare_adjoint_derivative_pw_sum_flux.py",
    #note above: the finite derivative does not seem to converge for every cell ids
    #so we only took some which converge
    # TODO "Adjoint Total Derivative with Filter" : "",
    "Run Optimization Implicit Grid" : "test_optimization/make_optimization_imp.py",
    "Run Optimization Explicit Grid" : "test_optimization/make_optimization_exp.py",
    "Density Filter" : "test_filter/test_density_filter.py",
    "Heavyside Density Filter" : "test_filter/test_heavyside_filter.py"
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
    log.write(f"Script \"{src_file}\"\n")
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
  print("See \"verification.log\" file for detailed informations\n")
  log = open("verifications.log",'a')
  log.write("\n=============================\n")
  log.write(f"\nTests conducted: {len(tests)}\n")
  log.write(f"Passed: {success}\n")
  log.write(f"Failed: {error}")
  log.close()
  return


if __name__ == "__main__":
  make_verifications()
