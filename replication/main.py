"""Main entry point for DOTs-SOCP.
"""

import sys
from pathlib import Path
root_of_import = Path(__file__).parent.parent
if str(root_of_import) not in sys.path:
    sys.path.insert(0, str(root_of_import))

from dot_surface_socp import solver as dot_solver
from dot_surface_socp import print_example_info
from dot_surface_socp import set_logging_level
from dot_surface_socp import run_dot_surface, run_dot_surface_versus_exact
from dot_surface_socp import parse_args as parse_args_socp


if __name__ == "__main__":
	args = parse_args_socp()
	set_logging_level(log_level=args.log_level, log_file=args.log_file)
	print_example_info(args)

	if not getattr(args, "versus_exact", False):
		run_dot_surface(
			solver=dot_solver,
			solver_name="socp",
			opts=args
		)
	else:
		run_dot_surface_versus_exact(
			solver=dot_solver,
			solver_name="socp",
			opts=args
		)
