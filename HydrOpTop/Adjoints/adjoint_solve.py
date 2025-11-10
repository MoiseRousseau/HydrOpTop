import time
from scipy.io import mmwrite
import scipy.sparse.linalg as spla
from scipy.sparse import dia_matrix
import scipy.sparse as sp
from scipy.sparse.csgraph import reverse_cuthill_mckee
from scipy.sparse.csgraph import maximum_bipartite_matching
import numpy as np




class Direct_Sparse_Linear_Solver:
    def __init__(self, algo="lu"):
        """

        """
        self.algo = algo.lower()
        if self.algo == "pardiso":
            try:
                import pypardiso
            except:
                raise RuntimeError("Please install pypardiso library to use PARDISO direct solver.")
        self.solve_funcs = {
            "lu": self.__lu_solve__,
            "spsolve": self.__sp_solve__,
            "pardiso": self.__pardiso_solve__,
        }
        self.solve_func = self.solve_funcs.get(self.algo,None)
        if self.solve_func is None:
            raise RuntimeError(f"Direct solver not recognized, try one of {[x for x in self.solve_funcs.keys()]}")
        return

    def __sp_solve__(self, A, b):
        """
        A simple wrapper around SciPy spsolve routine
        """
        l = spla.spsolve(A.tocsr(), b)
        return l

    def __lu_solve__(self, A, b):
        """
        LU direct solver from SciPy library.
        If LU solve is impossible, scale diagonal to force a solution
        """
        damped_factor = [0.,1e-12,1e-8,1e-6,1e-4,1e-3,1e-2,1e-1,1.]
        for df in damped_factor:
            if df != 0.:
                print(f"    Current damping factor: {df}")
            #_A = ( A + sp.diags(A.diagonal()*df) ).tocsc()
            _A = ( A + sp.eye(A.shape[0]) * df * A.diagonal().mean() ).tocsc()
            try:
                LU = spla.splu(_A)
                l = LU.solve(b)
                break
            except Exception as e:
                print(f"LU solve failed ({e}), try increasing damping factor")
        return l

    def __pardiso_solve__(self, A, b):
        """
        Solve large sparse linear systems of equations with the Intel oneAPI Math Kernel Library PARDISO solver.
        Use a simple wrapper around pypardiso spsolve routine.
        """
        x = pypardiso.spsolve(A,b)
        return x
    
    def jacobi_row_scaling(self, A, b, power=0.5):
        scale = np.power(np.abs(A.diagonal()), power)
        scale[scale == 0] = 1.0
        D_inv = sp.diags(1.0 / scale)
        A_s = D_inv @ A @ D_inv
        b_s = D_inv @ b
        return A_s, b_s, D_inv

    def solve(self, A, b):
        print(f"Solve adjoint equation using {self.algo} solver")
        t_start = time.time()
        A_scaled, b_scaled, D_inv = self.jacobi_row_scaling(A, b, power=0.5)
        l_scaled = self.solve_func(A_scaled,b_scaled)
        l = D_inv @ l_scaled
        print(f"Time to solve adjoint: {(time.time() - t_start)} s")
        return l


class Iterative_Sparse_Linear_Solver:
    def __init__(self, algo="bicgstab", preconditionner=None, reorder=True, verbose=True, solver_kwargs=None):
        """
        Solve A x = b using an iterative algorithm from SciPy library.

        Optionally apply Reverse Cuthill–McKee (RCM) reordering first for
        better sparse matrix vector product performance.

        Also scale matrix using Jacobi preconditionner and further improve the
        matrix condition number using Ruiz equilibration.

        Parameters
        ----------
        algo : str
            Algorithm to solve the iterative problem among ["bicgstab" (default), "gmres", "lsqr"]
        preconditionner : str
            The preconditionner type among ["ilu" (default), "diagonal"]
        reorder : bool
            If True, apply Reverse Cuthill–McKee reordering before ILU.
        verbose : bool
            Print progress information.
        solver_kwargs : dict
            Argument to pass to the SciPy function

        Returns
        -------
        x : ndarray
            Solution vector.
        """
        self.algo = algo.lower()
        self.reorder = reorder
        self.verbose = verbose
        self.solver_kwargs = solver_kwargs
        self.preconditionner = preconditionner
        self.l0 = None #previous solution
        self.perm_rcm = None
        self.perm_type = "rcm"
        # default value if not supplied
        if  self.solver_kwargs is None:
            self.solver_kwargs = {"rtol":3e-3,"atol":1e-40,"maxiter":400}
        return
    
    def permute_rcm(self, A):
        if self.perm_rcm is None:
            self.perm_rcm = reverse_cuthill_mckee(A, symmetric_mode=True)
        return self.perm_rcm

    def permute_large_to_diagonal(self, A):
        """
        Permute rows/columns of square sparse matrix A so that large entries
        are placed (as far as possible) on the diagonal.

        Returns:
            perm_r  : row permutation (array of size n)
            perm_c  : column permutation (array of size n)
        """
        n, m = A.shape
        assert n == m, "Only square matrices supported in this routine"
        # Build bipartite graph: rows i, columns j, weight = |A[i,j]|
        # Represented by sparse matrix abs(A)
        A_abs = abs(A)
        # For unweighted matching
        # Build bipartite adjacency (rows->cols if nonzero) and use maximum_bipartite_matching
        match = maximum_bipartite_matching(A_abs, perm_type='row')
        # match is array of size n: for each row i, match[i] = column matched or -1
        perm_r = np.arange(n)
        perm_c = np.array([ match[i] if match[i]>=0 else i for i in range(n) ], dtype=int)
        # Permutations: we want to permute A so that row i maps to perm_r[i], col j maps to perm_c[j].
        # Actually we want P A Q where P and Q are permutation matrices.
        # A_perm = A[perm_r, :][:, perm_c]
        return perm_r, perm_c


    def callback_scipy(self,xk,A,b):
        if self.callback_it % 50 == 0 and self.verbose:
            res = np.linalg.norm(b - A @ xk) / np.linalg.norm(b)
            print(f"Iter {self.callback_it}: residual = {res:.4e}")
        self.callback_it += 1
        return

    def jacobi_row_scaling(self, A, b, power=0.5):
        scale = np.power(np.abs(A.diagonal()), power)
        scale[scale == 0] = 1.0
        D_inv = sp.diags(1.0 / scale)
        A_s = D_inv @ A @ D_inv
        b_s = D_inv @ b
        return A_s, b_s, D_inv

    def ruiz_equilibrate(self, A, b, max_iter=5, tol=1e-4, eps=1e-8, verbose=False):
        """
        Perform Ruiz iterative equilibration on sparse matrix A and rhs b to improve matrix condition number.
        From Ruiz (2017), "A scaling algorithm to equilibrate both rows and columns norms in matrices"
        https://cerfacs.fr/wp-content/uploads/2017/06/14_DanielRuiz.pdf
        """
        m, n = A.shape
        D_r = np.ones(m)
        D_c = np.ones(n)
        b_eq = b.copy()
        for k in range(max_iter):
            # Row normalization
            row_norm = np.abs(A.tocsr()).max(axis=1).toarray().ravel()
            row_norm[row_norm < eps] = 1.0
            r_scale = 1.0 / np.sqrt(row_norm)
            A = sp.diags(r_scale) @ A
            b_eq *= r_scale
            D_r *= r_scale

            # Column normalization
            col_norm = np.abs(A.tocsr()).max(axis=0).toarray().ravel()
            col_norm[col_norm < eps] = 1.0
            c_scale = 1.0 / np.sqrt(col_norm)
            A = A @ sp.diags(c_scale)
            D_c *= c_scale

            test = np.max([np.abs(1 - np.max(row_norm)), np.abs(1 - np.max(col_norm))])
            if verbose:
                print(f"Ruiz iter {k+1} convergence test: {test} < {tol} ?")
            if test < tol:
                break

        return A, b_eq, D_r, D_c

    def solve(self, A, b):

        print(f"Solve adjoint equation using {self.algo} solver")
        self.callback_it = 0
        t_start = time.time()

        # Ensure A is sparse and CSC
        A = A.tocsc()

        # --- Step 1: optional Reverse Cuthill–McKee reordering ---
        if self.reorder:
            if self.perm_type == "rcm":
                if self.verbose:
                    print("Applying Reverse Cuthill–McKee reordering...")
                perm_r = self.permute_rcm(A)
                A_perm = A[perm_r, :][:, perm_r]
                b_perm = b[perm_r]
                perm_c = perm_r
                if self.l0 is not None: self.l0 = self.l0[perm_r]
            elif self.perm_type == "ltd":
                perm_r, perm_c = self.permute_large_to_diagonal(A)
                A_perm = A[perm_r, :][:, perm_c]
                b_perm = b[perm_r]
        else:
            perm = np.arange(A.shape[0])
            A_perm, b_perm = A, b

        # --- Step 2: Preconditioner ---
        # Always do a row scaling before anything
        A_scaled, b_scaled, D_inv = self.jacobi_row_scaling(A_perm, b_perm, power=0.5)
        A_scaled, b_scaled, Dr, Dc = self.ruiz_equilibrate(
            A_scaled, b_scaled, max_iter=6, verbose=self.verbose
        )
        D_inv = D_inv @ sp.diags(Dc)
        # Then preconditionning
        if self.preconditionner == "ilu":
            try:
                ilu = spla.spilu(A_scaled.tocsc(), drop_tol=1e-4, fill_factor=1.0)
                M = spla.LinearOperator(A_scaled.shape, ilu.solve)
            except RuntimeError as e:
                print(f"ILU failed: {e}. Continue with Diagonal preconditionner only")
                M = None
                #M = dia_matrix((np.sqrt(1.0 / np.abs(A.diagonal())), [0]), shape=A.shape)
        else:
            # no preconditionner
            M = None

        # --- Step 3: solve ---
        callback = lambda xk: self.callback_scipy(xk, A_scaled, b_scaled)

        if self.verbose:
            print("Starting iterative solve...")

        if self.algo == "bicgstab":
            x_scaled, info = spla.bicgstab(
                A_scaled, b_scaled, x0=self.l0, M=M, callback=callback,
                **self.solver_kwargs,
                #callback_type='x',
            )
        elif self.algo == "bicg":
            x_scaled, info = spla.bicg(
                A_scaled, b_scaled, x0=self.l0, M=M, callback=callback,
                **self.solver_kwargs,
            )
        elif self.algo == "gmres":
            x_scaled, info = spla.gmres(
                A_scaled, b_scaled, x0=self.l0, M=M, callback=callback,
                callback_type='x',
                **self.solver_kwargs,
            )
        elif self.algo == "lsqr":
            # scale by hand
            x_scaled = spla.lsqr(
                A_scaled, b_scaled, x0=self.l0,
                show=self.verbose,
                **self.solver_kwargs,
            )
        else:
            raise RuntimeError(f"Iterative algorithm {self.algo} not found...")

        # Approximate a-fortiori condition number
        if self.verbose:
            # https://cs357.cs.illinois.edu/textbook/notes/condition.html
            conds = []
            for k in range(5):
                perturb = np.random.random(len(b_scaled))-0.5
                b_perturb = A @ (x_scaled + perturb) - b_scaled
                cond_num = np.linalg.norm(perturb) / np.linalg.norm(x_scaled)
                cond_den = np.linalg.norm(b_perturb) / np.linalg.norm(b_scaled)
                conds.append(cond_den / cond_num)
            print(f"A-fortiori estimation of matrix condition number: {np.mean(conds):.4e} +/- {np.std(conds):.4e}") #REVERSED COMPARED TO PAPER

        # --- Step 4: Undo permutation and row scaling ---
        x_perm = D_inv @ x_scaled
        x = np.zeros_like(x_perm)
        x[perm_c] = x_perm

        if info == 0:
            if self.verbose:
                res = np.linalg.norm(b_scaled - A_scaled @ x_scaled) / np.linalg.norm(b_scaled)
                print(f"✅ Converged successfully within {self.callback_it} iterations and final residual {res:4e}.")
        elif info > 0:
            print(f"⚠️ {self.algo} Reached maximum iterations ({info}).")
        else:
            print("❌ Solver failed.")

        if self.verbose:
            print(f"True unscaled residual {np.linalg.norm(b - A @ x) / np.linalg.norm(b)}.")

        # if self.verbose:
        #     print(f"Final residual: {np.linalg.norm(b - A @ x):.2e}")
        #     print(f"Total iterations: {self.callback_it}")

        self.l0 = x_perm.copy()

        print(f"Time to solve adjoint: {(time.time() - t_start)} s")
        return x