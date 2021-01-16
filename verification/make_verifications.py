
import subprocess

def make_verifications():
  tests = {
  "Adjoint Derivative vs Finite Difference" : "adjoint_derivative/compare_adjoint_derivative.py"
  }
  
  success = 0
  error = 0
  print("\nHydrOpTop verification tests\n")
  for name,src_file in tests.items():
    print(f"{name}: ",end="",flush=True)
    folder = '/'.join(src_file.split('/')[:-1])
    script = src_file.split('/')[-1]
    cmd = ["python3",script]
    ret = subprocess.call(cmd, stdout=open("verifications.log",'w+'),
                          cwd=folder)
    if ret == 0: 
      print("Success")
      success += 1
    else: 
      print("ERROR!")
      error += 1
  
  print(f"\nTests conducted: {len(tests)}")
  print(f"Passed: {success}")
  print(f"Failed: {error}\n")
  return


if __name__ == "__main__":
  make_verifications()
