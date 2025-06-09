"""Demonstration of the usage.
"""

from dot_surface_socp import print_example_info
from dot_surface_socp import set_logging_level
from dot_surface_socp import run_dot_surface
from dot_surface_socp import parse_args as parse_args_socp

# Default choices
DEFAULT_EXAMPLE = "airplane" # ["airplane", "armadillo", "ring", "knots_3", "knots_5", ...]
DEFAULT_TOL = 1e-3 # Default tolerance for the solver
DEFAULT_ITERATIONS = 1000 # Default maximal number of iterations for the solver

def parse_args():
    """Parse command line arguments for the script.
    """
    parser = parse_args_socp(return_parser=True)
    parser._option_string_actions['--example'].default = DEFAULT_EXAMPLE
    parser._option_string_actions['--show'].default = True
    parser._option_string_actions['--tol'].default = DEFAULT_TOL
    parser._option_string_actions['--nit'].default = DEFAULT_ITERATIONS

    return parser.parse_args()

# Print example information
args = parse_args()
set_logging_level(log_level=args.log_level, log_file=args.log_file)
print_example_info(args)

# Run the solver
run_dot_surface(
    solver_name="socp",
    opts=args
)
