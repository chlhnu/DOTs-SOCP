"""Evalulate the iterative solution versus the exact transportation at some checkpoints during iteration.

It gives a comparison of the kkt error and the error versus the exact solution.
"""

import sys
from pathlib import Path
root_of_import = Path(__file__).parent.parent
if str(root_of_import) not in sys.path:
    sys.path.insert(0, str(root_of_import))

from dot_surface_socp import solver as dot_solver
from dot_surface_socp import run_dot_surface_versus_exact
from dot_surface_socp import print_example_info
from dot_surface_socp import set_logging_level
from dot_surface_socp import parse_args as parse_args_socp
from math import log

def parse_args(return_parser=False):
    """Parse command line arguments for the script.
    """
    parser = parse_args_socp(return_parser=True)
    
    # Restrict example argument to one that has defined the exact transportation
    parser._option_string_actions['--example'].choices = ["plane"]
    parser._option_string_actions['--example'].help = \
        "Example to solve.\n" \
        "Require function definition of 'get_exact_transportation' in the setting file."
    
    # Remove versus_exact argument (which is a necessary argument in this file)
    parser._actions = [action for action in parser._actions if action.dest != 'versus_exact']
    parser._option_string_actions.pop('--versus_exact', None)

    # Modify outdir argument
    parser._option_string_actions['--outdir'].default = "output/undated_versus_exact"
    
    if return_parser:
        return parser
    else:
        return parser.parse_args()
    

def automatic_checkpoints(tol: float):
    """Generate tol checkpoints
    """
    negative_exponents_raw = - log(tol, 10)
    negative_exponents = int(
        round(negative_exponents_raw, 12) if abs(negative_exponents_raw - round(negative_exponents_raw)) < 1e-12 else negative_exponents_raw
    )
    return [10 ** (- i - 1) for i in range(negative_exponents)]


if __name__ == "__main__":
    args = parse_args()
    set_logging_level(log_level=args.log_level, log_file=args.log_file)
    print_example_info(args)

    if not args.checkpoints:
        args.checkpoints = automatic_checkpoints(args.tol)

    run_dot_surface_versus_exact(
        solver=dot_solver,
        solver_name="socp",
        opts=args
    )
