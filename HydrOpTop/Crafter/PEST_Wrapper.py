"""
Wrapper to PEST suite
"""

import numpy as np
import dill
import subprocess
import datetime
import os, shutil
from tempfile import TemporaryDirectory


class PEST_Wrapper:
    """
    A Python Wrapper around PEST
    """

    def __init__(self,
                 NPAR, NOBS,
                 NOPTMAX: int = 10,
                 PHIREDSTP: float = 5e-3,
                 NPHISTP: int = 3,
                 NPHINORED: int = 4,
                 RELPARSTP: float = 5e-3,
                 NRELPAR: int = 1,
                 RLAMBDA1: float = 10.0,
                 PHIREDSWH: float = 0.05,
                 workers: int = 1,
                 PEST_executable: str = "pestpp-glm",
                 files_to_copy: list = [],
                 create_file_only: bool = False,
                 PEST_port = 55315,
                 rundir: str = "",
                 create_safe_wrapper: bool = False,
                 verbose: bool = False):
        
        """
        :param PEST_executable: Command to launch PEST or PESTpp program
        :type PEST_executable: str
        """

        self.NOPTMAX = NOPTMAX
        self.NPAR = NPAR
        self.NOBS = NOBS
        self.PHIREDSTP = PHIREDSTP
        self.NPHISTP = NPHISTP
        self.NPHINORED = NPHINORED
        self.RELPARSTP = RELPARSTP
        self.NRELPAR = NRELPAR

        self.RLAMBDA1 = RLAMBDA1

        self.PHIREDSWH = PHIREDSWH

        if rundir: self.tmpdir = rundir

        self.workers = workers
        self.PEST_executable = PEST_executable
        self.PEST_port = PEST_port
        self.files = files_to_copy
        self.create_file_only = create_file_only
        self.create_safe_wrapper = create_safe_wrapper
        self.verbose = verbose
        return

    # ---------------------------------------------------
    def fit(self, rfun, x0, bounds=None, jac=None):
        # Create PEST file
        self._write_pst(x0, bounds, dir=self.tmpdir)
        self._write_ins(self.tmpdir)
        self._write_tpl(self.tmpdir)
        with open(self.tmpdir+"/run_model.py","w") as f:
            f.write("import dill\n")
            f.write("with open('model.pkl','rb') as f:\n")
            f.write("    dill.load(f)()\n")
        # write safe wrapper
        if self.create_safe_wrapper:
            with open(self.tmpdir+"/run_model_safe.sh","w") as f:
                f.write("#!/bin/bash\n\n")
                f.write("python run_model.py\n")
                f.write("status=$?\n\n")
                f.write("if [ $status -ne 0 ]; then\n")
                f.write("    echo \"run_model.py failed with status $status\" >&2\n")
                f.write("fi\n\n")
                f.write("exit 0")

        # Dump the rfunc so it can be call from command line
        l = lambda: np.savetxt("./outputs.txt", rfun(np.loadtxt("./inputs.txt")))
        with open(self.tmpdir+"/model.pkl", "wb") as f:
            dill.dump(l, f)

        # Run PEST
        ret = None
        if self.create_file_only:
            return
        if self.workers > 1:
            ret = subprocess.run(
                [self.PEST_executable, "pest.pst", "/H", f":{self.PEST_port}"],
                cwd=self.tmpdir
            )
            for i in range(self.workers):
                subprocess.run([self.PEST_executable, "pest.pst", "/H", f"localhost:{self.PEST_port}"])
        else:
            ret = subprocess.run(
                [self.PEST_executable, "pest.pst"],
                cwd=self.tmpdir
            )
        if ret.returncode != 0:
            raise RuntimeError("Non zero return code from PEST")

        # Get results
        x_opt = np.loadtxt(self.tmpdir+"/pest.par", skiprows=1, usecols=[1])
        func_eval_history = np.loadtxt(self.tmpdir+"/pest.iobj", skiprows=1, usecols=[1,2], delimiter=",")
        func_eval_history[:,1] = np.sqrt(func_eval_history[:,1])
        residuals = np.loadtxt(self.tmpdir+"/pest.rei", skiprows=4, usecols=[3])
        return self._finish(x_opt, True, "", len(func_eval_history), func_eval_history, [])

    # ---------------------------------------------------
    def _finish(self, p, success, msg, it, func_eval_history, x_hist):
        return {
            "x": p,
            "success": success,
            "message": msg,
            "niter": it,
            "func_eval_history": func_eval_history,
            "x_history": x_hist
        }

    # ---------------------------------------------------
    def _write_pst(self, x0, bounds, dir='.'):
        # create the pst file
        pst_out = ["pcf"]
        pst_out.append("* control data")
        pst_out.append("norestart estimation")
        pst_out.append(f"{self.NPAR} {self.NOBS} 1 0 1")
        pst_out.append("1 1 double point 1 0 0")
        pst_out.append(f"{self.RLAMBDA1} 2.0 0.3 0.03 10 0 lamforgive derforgive")
        pst_out.append(f"4. 4. 0.001")
        pst_out.append(f"{self.PHIREDSWH}")
        pst_out.append(f"{self.NOPTMAX} {self.PHIREDSTP} {self.NPHISTP} {self.NPHINORED} {self.RELPARSTP} {self.NRELPAR}")
        pst_out.append("0 0 0")

        pst_out.append("* singular value decomposition")
        pst_out.append("1")
        pst_out.append(f"{self.NPAR} 1.e-6")
        pst_out.append("1")

        pst_out.append("* parameter groups")
        pst_out.append("params relative 1e-4 0. switch 1.  parabolic")
        pst_out.append("* parameter data")
        for i in range(self.NPAR):
            pst_out.append(f"par{i} none factor {x0[i]} {bounds[i][0]} {bounds[i][1]} params 1. 0. 1")

        pst_out.append("* observation groups")
        pst_out.append("obs")
        pst_out.append("* observation data")
        for i in range(self.NOBS):
            pst_out.append(f"obs{i} 0. 1. obs")

        pst_out.append("* model command line")
        if self.create_safe_wrapper:
            pst_out.append("./run_model_safe.sh")
        else:
            pst_out.append("python run_model.py")
        pst_out.append("* model input/output")
        pst_out.append("params.tpl ./inputs.txt")
        pst_out.append("obs.ins ./outputs.txt")

        out = open(dir + "/pest.pst",'w')
        out.write("\n".join(pst_out))
        out.close()

    def _write_ins(self, dir):
        """
        Write PEST instruction file to read output of the solver
        """
        out = open(dir+"/obs.ins",'w')
        out.write("pif %\n")
        for i in range(self.NOBS):
            out.write(f"l1 !obs{i}!\n")
        out.close()
        return

    def _write_tpl(self, dir):
        """
        Write PEST template file to write input of the solver
        """
        out = open(dir+"/params.tpl",'w')
        out.write("ptf %\n")
        for i in range(self.NPAR):
            out.write(f"%par{i} %\n")
        out.close()
        return