import time
from scipy.io import mmwrite
import scipy.sparse.linalg as spla
from scipy.sparse import dia_matrix
import scipy.sparse as sp
from scipy.sparse.csgraph import reverse_cuthill_mckee
from scipy.sparse.csgraph import maximum_bipartite_matching
import numpy as np


def jacobi_row_scaling(A, b, power=0.5):
    scale = np.power(np.abs(A.diagonal()), power)
    scale[scale == 0] = 1.0
    D_inv = sp.diags(1.0 / scale)
    A_s = D_inv @ A @ D_inv
    b_s = D_inv @ b
    return A_s, b_s, D_inv

def ruiz_equilibrate(A, b, max_iter=5, tol=1e-4, eps=1e-8, verbose=False):
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
        b_eq *= r_scale[:,None] if b_eq.ndim == 2 else r_scale
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


class SSORPreconditioner:
    """
    Symmetric Successive Over-Relaxation (SSOR) preconditioner
    for a sparse matrix A.

    M^{-1} = (D + w L)^{-1} * D * (D + w U)^{-1}
    """

    def __init__(self, A, omega=1.0):
        if not sp.issparse(A):
            raise ValueError("A must be a SciPy sparse matrix")
        self.A = A.tocsr()
        self.n = A.shape[0]
        self.omega = omega

        # Extract D, L, U
        self.D = sp.diags(A.diagonal())
        self.L = sp.tril(A, -1)
        self.U = sp.triu(A, 1)

        # Build the two triangular SSOR factors
        self.M1 = self.D + omega * self.L    # Lower factor
        self.M2 = self.D + omega * self.U    # Upper factor

    def solve(self, b):
        """
        Apply M^{-1} * b using two triangular solves.
        """
        # Solve (D + w L) y = b
        y = spla.spsolve_triangular(self.M1, b, lower=True)
        # Scale with D
        y *= self.A.diagonal()
        # Solve (D + w U) x = y
        x = spla.spsolve_triangular(self.M2, y, lower=False)
        return x

    def as_linear_operator(self):
        """
        Return a scipy.sparse.linalg.LinearOperator to use in Krylov methods.
        """
        return spla.LinearOperator(
            shape=(self.n, self.n),
            matvec=self.solve,
            dtype=self.A.dtype
        )



class Direct_Sparse_Linear_Solver:
    def __init__(self, algo="lu", row_scaling=False, debug_write_matrix=False, verbose=False, **kwargs):
        """
        Solve A x = b using a direct algorithm, can handle multiple right hand
        side at the same time.

        Optionally apply Reverse Cuthill–McKee (RCM) reordering first for
        better sparse matrix vector product performance.

        Also scale matrix using Jacobi preconditionner and further improve the
        matrix condition number using Ruiz equilibration.
        """
        self.row_scaling = row_scaling
        self.debug = debug_write_matrix
        self.verbose = verbose
        self.n_solve = 0
        self.algo = algo.lower()
        self.pardiso_solver = None
        if self.algo == "pardiso":
            try:
                import pypardiso
            except:
                raise RuntimeError("Please install pypardiso library to use PARDISO direct solver.")
        if self.algo == "qr":
            try:
                import sparseqr
            except:
                raise RuntimeError("Please install sparse library to use QR factorisation and solver.")

        self.solve_funcs = {
            "lu": self.__lu_solve__,
            "qr": self.__qr_solve__,
            "umfpack": self.__umfpack_solve__,
            "pardiso": self.__pardiso_solve__,
        }
        self.solve_func = self.solve_funcs.get(self.algo,None)
        if self.solve_func is None:
            raise RuntimeError(f"Direct solver not recognized, try one of {[x for x in self.solve_funcs.keys()]}")
        return

    def __umfpack_solve__(self, A, b):
        """
        A simple wrapper around SciPy spsolve routine
        """
        try:
            from scikits.umfpack import spsolve as umf_spsolve
        except ImportError:
            raise RuntimeError("scikit-umfpack not installed")
        l = umf_spsolve(A, b)
        return l

    def __lu_solve__(self, A, b):
        """
        LU direct solver from SciPy library.
        If LU solve is impossible, scale diagonal to force a solution
        """
        LU = spla.splu(A)
        l = LU.solve(b)
        return l

    def __pardiso_solve__(self, A, b):
        """
        Solve large sparse linear systems of equations with the Intel oneAPI Math Kernel Library PARDISO solver.
        Use a simple wrapper around pypardiso spsolve routine.
        """
        try:
            from pypardiso import spsolve as pardiso_spsolve
            from pypardiso import PyPardisoSolver
        except ImportError:
            raise RuntimeError("pypardiso not installed")
        if self.pardiso_solver is None:
            self.pardiso_solver = PyPardisoSolver()
        x = pardiso_spsolve(A, b, solver=self.pardiso_solver)
        return x

    def __qr_solve__(self, A, b):
        import sparseqr
        x = sparseqr.solve(A, b, tolerance=0)
        return x

    def solve(self, A, b):
        self.n_solve += 1
        if self.verbose:
            print(f"Solve adjoint equation using {self.algo} solver")
            t_start = time.time()
        D_inv = 1.
        if self.row_scaling:
            A_scaled, b_scaled, D_inv = jacobi_row_scaling(A, b, power=0.5)
            A_scaled, b_scaled, Dr, Dc = ruiz_equilibrate(
                 A_scaled, b_scaled, max_iter=6, verbose=True
            )
            D_inv = D_inv @ sp.diags(Dc)
        else:
            A_scaled, b_scaled = A,b
        if self.debug:
            from scipy.io import mmwrite
            print("Write debug matrices")
            mmwrite("lhs.mtx", A_scaled)
            np.savetxt("rhs.csv", b_scaled)

        damped_factor = [0.,1e-6,1e-4,1e-3,1e-2,1e-1,1.]
        l_scaled = None
        for df in damped_factor:
            if df != 0.:
                print(f"    Current damping factor: {df}")
            #_A = ( A + sp.diags(A.diagonal()*df) ).tocsc()
                _A = ( A_scaled + sp.eye(A_scaled.shape[0]) * df * np.abs(A_scaled.diagonal()).mean()).tocsc()
            else:
                _A = A_scaled
            try:
                l_scaled = self.solve_func(_A,b_scaled)
                break
            except Exception as e:
                print(f"LU solve failed ({e}), try increasing damping factor")
        if l_scaled is None:
            raise RuntimeError("LU solve failed")
        l = D_inv @ l_scaled if self.row_scaling else l_scaled
        if self.verbose: print(f"Time to solve adjoint: {(time.time() - t_start)} s")
        return l

    def free_memory(self):
        if self.pardiso_solver: self.pardiso_solver.free_memory()
        return


class Iterative_Sparse_Linear_Solver:
    def __init__(self, algo="bicgstab", preconditionner="", row_scaling=True, reorder=True, verbose=False, debug_write_matrix=False, **kwargs):
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
        self.debug = debug_write_matrix
        self.preconditionner = preconditionner
        self.row_scaling = row_scaling
        self.l0 = None #previous solution
        self.perm_rcm = None
        self.perm_type = "rcm"
        # default value if not supplied
        self.solver_kwargs = {"rtol":3e-4,"atol":1e-40,"maxiter":250}
        self.solver_kwargs.update(kwargs)
        self.outer_v = [] #for lgmres recycling
        self.n_solve = 0
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


    def solve(self, A, b):

        if np.any(np.isnan(A.data)) or np.any(np.isnan(b)):
            raise RuntimeError("Adjoint system contains NaN. This is a HydrOpTop bug, please contact support.")

        if self.verbose:
            print(f"Solve adjoint equation using {self.algo} solver")
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
                if self.l0 is not None: self.l0 = self.l0[perm_r].T
            elif self.perm_type == "ltd":
                perm_r, perm_c = self.permute_large_to_diagonal(A)
                A_perm = A[perm_r, :][:, perm_c]
                b_perm = b[perm_r]
        else:
            perm = np.arange(A.shape[0])
            A_perm, b_perm = A, b

        # --- Step 2: Preconditioner ---
        # Always do a row scaling before anything
        if self.row_scaling:
            A_scaled, b_scaled, D_inv = jacobi_row_scaling(A_perm, b_perm, power=0.5)
            A_scaled, b_scaled, Dr, Dc = ruiz_equilibrate(
                A_scaled, b_scaled, max_iter=6, verbose=self.verbose
            )
            D_inv = D_inv @ sp.diags(Dc)
        # Then preconditionning
        if "ilu" in self.preconditionner:
            fill_factor = float(self.preconditionner[3:]) + 1.
            if self.verbose:
                print(f"ILU factorization for {self.algo} preconditionning, fill factor = {fill_factor}")
            M = None
            diag_shift = [0., 1e-12, 1e-8, 1e-6, 1e-4, 1e-3, 1e-2, 1e-1, 1]
            for alpha in diag_shift:
                if M is not None: break
                try:
                    ilu = spla.spilu(
                        (A_scaled + alpha * sp.eye(A_scaled.shape[0])).tocsc(),
                        drop_tol=1e-6, fill_factor=fill_factor
                    )
                    M = spla.LinearOperator(A_scaled.shape, ilu.solve)
                except RuntimeError as e:
                    print(f"ILU failed: {e}. Try increasing diagonal shift (current: {alpha})")
                    #M = dia_matrix((np.sqrt(1.0 / np.abs(A.diagonal())), [0]), shape=A.shape)
        elif self.preconditionner == "ssor":
            M = SSORPreconditioner(A_scaled).as_linear_operator()
        else:
            # no preconditionner
            M = None

        # --- Step 3: solve ---
        if self.verbose:
            print("Starting iterative solve...")

        if self.debug:
            print("Write debug matrices")
            from scipy.io import mmwrite
            mmwrite("lhs.mtx", A_scaled)
            np.savetxt("rhs.csv", b_scaled)

        RHS = b_scaled.T if b_scaled.ndim == 2 else [b_scaled]
        x_scaled = []
        for j,rhs in enumerate(RHS):
            if self.verbose:
                print(f"Solving for RHS {j}")
            self.callback_it = 0
            callback = lambda xk: self.callback_scipy(xk, A_scaled, rhs)
            l0 = self.l0[j] if self.l0 is not None else None
            if self.algo == "bicgstab":
                X, info = spla.bicgstab(
                    A_scaled, rhs, x0=l0, M=M, callback=callback,
                    **self.solver_kwargs,
                    #callback_type='x',
                )
                x_scaled.append(X)
            elif self.algo == "bicg":
                X, info = spla.bicg(
                    A_scaled, rhs, x0=l0, M=M, callback=callback,
                    **self.solver_kwargs,
                )
                x_scaled.append(X)
            elif self.algo == "gmres":
                X, info = spla.gmres(
                    A_scaled, rhs, x0=l0, M=M, callback=callback,
                    callback_type='x',
                    **self.solver_kwargs,
                )
                x_scaled.append(X)
            elif self.algo == "lgmres":
                X, info = spla.lgmres(
                    A_scaled, rhs, x0=l0, M=M, callback=callback,
                    #callback_type='x',
                    outer_v=self.outer_v,
                    outer_k=10,
                    **self.solver_kwargs,
                )
                x_scaled.append(X)
            elif self.algo == "lsqr":
                # scale by hand
                if "rtol" in self.solver_kwargs.keys():
                    self.solver_kwargs["atol"] = self.solver_kwargs["rtol"]
                    self.solver_kwargs["iter_lim"] = self.solver_kwargs["maxiter"]
                    self.solver_kwargs.pop('rtol')
                    self.solver_kwargs.pop('maxiter')
                X = spla.lsqr(
                    A_scaled, rhs, x0=l0,
                    show=self.verbose,
                    **self.solver_kwargs,
                )
                info = X[1]
                x_scaled.append(X[0])
            else:
                raise RuntimeError(f"Iterative algorithm {self.algo} not found...")

            if info == 0:
                if self.verbose:
                    res = np.linalg.norm(rhs - A_scaled @ X) / np.linalg.norm(rhs)
                    print(f"✅ Converged successfully within {self.callback_it} iterations and final residual {res:4e}.")
            elif info > 0:
                print(f"⚠️ {self.algo} Reached maximum iterations ({info}).")
            else:
                print("❌ Solver failed.")

        x_scaled = np.array(x_scaled).T if b_scaled.ndim == 2 else np.array(x_scaled).flatten()

        # Approximate a-fortiori condition number
        if self.verbose:
            # https://cs357.cs.illinois.edu/textbook/notes/condition.html
            conds = []
            for k in range(5):
                eps_pertub = 1e-3
                perturb = (np.random.random(b_scaled.shape)-0.5) * np.linalg.norm(x_scaled) * eps_pertub
                b_perturb = A @ (x_scaled + perturb) - b_scaled
                cond_num = np.linalg.norm(perturb) * eps_pertub
                cond_den = np.linalg.norm(b_perturb) / np.linalg.norm(b_scaled)
                conds.append(cond_den / cond_num)
            print(f"A-fortiori estimation of matrix condition number: {np.mean(conds):.4e} +/- {np.std(conds):.4e}") #REVERSED COMPARED TO PAPER

        # --- Step 4: Undo permutation and row scaling ---
        x_perm = D_inv @ x_scaled
        x = np.zeros_like(x_perm)
        x[perm_c] = x_perm

        if self.verbose:
            print(f"True unscaled residual {np.linalg.norm(b - A @ x) / np.linalg.norm(b)}.")

        # if self.verbose:
        #     print(f"Final residual: {np.linalg.norm(b - A @ x):.2e}")
        #     print(f"Total iterations: {self.callback_it}")

        self.l0 = x_perm.copy() if x_perm.ndim == 2 else x_perm.copy()[:,None]

        if self.verbose: print(f"Time to solve adjoint: {(time.time() - t_start)} s")
        return x