"""
This module implements a Second-Order Cone Programming (SOCP) solver for DOT on surface.
It includes functions for data preprocessing and solver execution with customizable parameters.
"""

from .solver_decorator import solver_raw
from .solver_decorator import solver

__all__ = [
    "solver_raw", # The SOCP solver, which returns the solution to SOCP formulation of DOT
    "solver", # The DOT solver, which will call the SOCP solver and return the solution to original DOT formulation
]