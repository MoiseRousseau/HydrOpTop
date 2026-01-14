# This file is a part of the topology optimization program by Moise Rousseau

import numpy as np
from typing import Any, Dict, Optional


class Base_Function:
    """
    Base class for defining a cost or objective function in a numerical optimization
    or simulation workflow.

    This class provides an interface for:
      - Setting and retrieving solver inputs
      - Managing adjoint problems
      - Mapping between parameter indices and cell identifiers
      - Evaluating cost functions and their derivatives
    """

    def __init__(self) -> None:
        self.name: str = "Base Function"
        self.cell_ids: Optional[np.ndarray] = None
        self.inputs: Dict[str, Any] = {}
        self.initialized: bool = False
        self.adjoint: Optional[Any] = None  # Adjoint variable (if provided by the problem crafter)
        self.p_ids: Optional[np.ndarray] = None
        self.ids_p: Optional[np.ndarray] = None
        self.variables_needed: Optional[list[str]] = None
        self.indexes = None #Define the required cell_ids data
        self.linear = False # Set this to true if the function is linear for better performance

    # -------------------------------------------------------------------------
    # Adjoint / Input Handling
    # -------------------------------------------------------------------------
    def set_adjoint_problem(self, adjoint: Any) -> None:
        """Attach an adjoint problem or variable to this function."""
        self.adjoint = adjoint

    def set_inputs(self, inputs: Dict[str, Any]) -> None:
        """
        Set the solver output variables required by this function.

        Note:
            This method is intended to be called once during initialization.
            The `inputs` dictionary will then be updated in place, so
            subsequent calls are unnecessary.
        """
        self.inputs = inputs

    def get_inputs(self) -> Dict[str, Any]:
        """Return the current input values."""
        return self.inputs

    # -------------------------------------------------------------------------
    # Parameter Mapping
    # -------------------------------------------------------------------------
    def set_p_to_cell_ids(self, p_ids: np.ndarray) -> None:
        """
        Define the mapping between parameter indices and cell identifiers.

        Args:
            p_ids (np.ndarray): Array where `p_ids[i]` is the cell ID corresponding
                to the i-th parameterized cell.

        Creates:
            - `self.p_ids`: forward mapping (parameter index → cell ID)
            - `self.ids_p`: reverse mapping (cell ID → parameter index, -1 if not parameterized)
        """
        self.p_ids = np.asarray(p_ids, dtype=np.int64)
        max_id = np.max(self.p_ids)
        self.ids_p = -np.ones(max_id, dtype=np.int64)  #-1 mean not optimized
        self.ids_p[self.p_ids - 1] = np.arange(len(self.p_ids))

    # -------------------------------------------------------------------------
    # Objective Evaluation
    # -------------------------------------------------------------------------
    def evaluate(self, p: np.ndarray) -> float:
        """
        Evaluate the cost or objective function.

        Args:
            p (np.ndarray): Current parameter vector.

        Returns:
            float: The scalar value of the cost function.
        """
        return 0.0

    # -------------------------------------------------------------------------
    # Derivatives
    # -------------------------------------------------------------------------
    def d_objective(self, var: str, p: np.ndarray) -> np.ndarray:
        """
        Compute the derivative of the objective with respect to a given variable.

        Args:
            var (str): Variable name.
            p (np.ndarray): Current parameter vector.

        Returns:
            np.ndarray: Partial derivative with respect to the variable.
        """
        eps = 1e-6
        if var in self.variables_needed:
            # Do finite difference
            res = np.zeros_like(self.inputs[var])
            v = self.inputs[var].copy()
            for i in range(len(res)):
                v[i] += eps
                f1 = self.evaluate(p)
                v[i] -= 2*eps
                f2 = self.evaluate(p)
                res[i] = (f2 - f1) / (2 * eps)
        else:
            res = np.zeros_like(p)
        return res

    def d_objective_dp_partial(self, p: np.ndarray) -> np.ndarray:
        """
        Compute the explicit derivative of the objective with respect to
        the parameter vector `p` (if it appears directly in the definition).

        Args:
            p (np.ndarray): Current parameter vector.

        Returns:
            np.ndarray: Partial derivative with respect to `p`.
        """
        return np.zeros_like(p)


    def output_to_user(self):
        return {}

    # -------------------------------------------------------------------------
    # Metadata Accessors
    # -------------------------------------------------------------------------
    def __get_variables_needed__(self) -> Optional[list[str]]:
        """Return the list of variable names required for computation."""
        return self.variables_needed

    def __get_name__(self) -> str:
        """Return the name of the function."""
        return self.name

