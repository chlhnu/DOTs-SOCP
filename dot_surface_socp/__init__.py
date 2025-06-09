"""DOTs-SOCP: SOCP for Dynamical Optimal Transport on Discrete Surface.

This package implements a Second-Order Cone Programming (SOCP) reformulation 
of the dynamical optimal transport problem on triangulated surfaces. It solves 
optimal transport between probability distributions on discrete triangulated 
meshes using an inexact ALM (iALM) algorithm for solving the SOCP reformulation.
"""

from .socp import solver_raw
from .socp import solver
from .interface import run_dot_surface
from .interface import run_dot_surface_versus_exact
from .interface import print_example_info
from .interface import set_logging_level
from .cli import parse_args

__all__ = [
    "solver_raw", # The SOCP solver, which returns the solution to SOCP formulation of DOT-Surface
    "solver", # The DOT-Surface solver, which will call the SOCP solver and return the solution to DOT-Surface
    "run_dot_surface", # The interface to run the DOT-Surface with given solver and options
    "run_dot_surface_versus_exact", # The interface to run the DOT-Surface with given solver and options, comparing against exact solution
    "print_example_info", # Print information of the current run
    "set_logging_level", # Set the logging level and file for the current run
    "parse_args", # Parser for command line arguments to call `run_dot_surface` or `run_dot_surface_versus_exact` with given options
]
