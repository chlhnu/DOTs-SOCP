"""Decorators for solver functions to standardize their output format.
"""

from typing import Tuple
import numpy as np
from dot_surface_socp.utils.type import SolutionDotData, GeometryData
from dot_surface_socp.utils.admm_tools import RunningHistory


def solver_decorator_socp_to_dot(solver_socp):
    """Decorator to convert SOCP solver output to standard format.
    
    The SOCP solver returns (SolutionSocpData, RunningHistory)
    This decorator returns a function whose returned result is (SolutionDotData, RunningHistory)
    """
    def solver_dot(n_time: int, geometry: GeometryData, **kwargs) -> Tuple[SolutionDotData, RunningHistory]:
        # Call the original solver
        solution_socp, run_history = solver_socp(n_time, geometry, **kwargs)
        
        # Convert to standard format
        from dot_surface_socp.utils.type import translate_solution_socp_to_dot
        solution_dot = translate_solution_socp_to_dot(solution_socp=solution_socp, geom=geometry)
        
        return solution_dot, run_history
    
    return solver_dot


def solver_decorator_time_stagger_to_center(solver_dot):
    """Decorator to convert DOT solver output to one located at time-centered discrete grid.
    """
    def __to_time_centered(solution_dot: SolutionDotData, mu0: np.ndarray, mu1: np.ndarray):
        mux = 0.5 * (solution_dot["mu"][:-1] + solution_dot["mu"][1:])
        solution_dot["mu"] = np.concatenate([mu0[None, :], mux, mu1[None, :]], axis=0)

    def solver_dot_center(n_time: int, geometry: GeometryData, **kwargs) -> Tuple[SolutionDotData, RunningHistory]:
        # Get the initial and final densities
        mu0 = geometry["mu0"]
        mu1 = geometry["mu1"]

        # Call the original solver
        solution_dot, run_history = solver_dot(n_time, geometry, **kwargs)

        # Convert to time-centered grid (not include checkpoints)
        __to_time_centered(solution_dot, mu0, mu1)

        # Convert checkpoints to time-centered grid
        if "checkpoints" in solution_dot and solution_dot["checkpoints"]:
            for checkpoint in solution_dot["checkpoints"]:
                __to_time_centered(checkpoint, mu0, mu1)

        return solution_dot, run_history
    
    return solver_dot_center


# Decorate the socp solver
from dot_surface_socp.socp.solver_socp import solver_socp
solver_raw = solver_decorator_socp_to_dot(solver_socp)
solver_raw.__name__ = "dot_solver_socp"
solver_raw.__doc__ = (
    "Solver function for the DOT problem.\n"
    "This function will solve the DOT problem using the SOCP solver.\n"
)

solver = solver_decorator_time_stagger_to_center(solver_raw)
solver.__name__ = "dot_solver_socp_center"
solver.__doc__ = (
    "Solver function for the DOT problem.\n"
    "This function will solve the DOT problem using the SOCP solver "
    "and return a solution located at the time-centered discrete grid.\n"
)
