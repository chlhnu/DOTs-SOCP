"""An easy-to-use command-line interface for DOT-Surface-SOCP.

Provides argument parsing and configuration for running DOT-Surface simulations with SOCP solver.
Supports various parameters for mesh selection, algorithm settings, and visualization options.
"""

import argparse
from numpy import inf as np_inf
from dot_surface_socp.interface import print_example_info
from dot_surface_socp.interface import set_logging_level
from dot_surface_socp.interface import run_dot_surface, run_dot_surface_versus_exact


class CheckArgsRangeTau(argparse.Action):
	def __call__(self, arg_parser, namespace, values, option_string=None):
		if not (0.0 < float(values) < 2.0):
			raise argparse.ArgumentError(self, f"tau must be in range (0.0, 2.0), but got {values}")
		setattr(namespace, self.dest, values)

class CheckArgsPowerPerceptual(argparse.Action):
    def __call__(self, arg_parser, namespace, values, option_string=None):
        if not float(values) > 0.0:
            raise argparse.ArgumentError(self, f"power_perceptual must be > 0.0, but got {values}")
        setattr(namespace, self.dest, values)
		

def parse_args(parser=None, return_parser=False):
    """Parse command line arguments.

    Parameters:
        parser (argparse.ArgumentParser): Optional
            Argument parser instance.
            if provided, add dot surface specific arguments to the existing parser.
    """
    if parser is None:
        parser = argparse.ArgumentParser(
            description="Animate DOT-Surface transmission process."
        )

    # Example
    example_group = parser.add_argument_group(
        "Example configuration",
        "Configure example via predefined set (--example) or custom files (--mesh_file, --setting_file)")
    example_group.add_argument(
        "--example", default=None, type=str,
        help='''Select pre-defined example to run: [
                    "airplane", "armadillo", "hand", "punctured_ball", "bunny",
                    "sphere", "ring", "knots_3", "knots_5", "hills"
                    "refined_airplane", "refined_armadillo", "refined_hand", "refined_punctured_ball", "refined_bunny"
                ]''')
    example_group.add_argument(
        "--mesh_file", default=None, type=str,
        help="Input mesh file path (.off)")
    example_group.add_argument(
        "--setting_file", default=None, type=str,
        help="Input settings file path (.py)")
    example_group.add_argument(
        "--congestion", default=0.0, type=float,
        help="Congestion factor (default: 0.0)")
    example_group.add_argument(
        "--ntime", default=31, type=int,
        help="Time discretization points")

    # Parameters of algorithm
    algo_params_group = parser.add_argument_group(
        "Algorithm parameters")
    algo_params_group.add_argument(
        "--tau", type=float, default=None, action=CheckArgsRangeTau,
        help="Step size value (range: 0.0-2.0)")
    algo_params_group.add_argument(
        "--nit", default=10**3, type=int,
        help="Maximum iterations")
    algo_params_group.add_argument(
        "--eps", default=0.0, type=float,
        help="Epsilon value for proximal item in the phi step")
    algo_params_group.add_argument(
        "--tol", default=10**(-3), type=float,
        help="Convergence tolerance")
    algo_params_group.add_argument(
        "--time_limit", default=np_inf, type=float,
        help="Time limit for the algorithm (in seconds, default: no time limit)")

    # Show or Save results and running history
    output_group = parser.add_argument_group(
        "Output options")
    output_group.add_argument(
        "--show", default=False, action='store_true',
        help="Display animation")
    output_group.add_argument(
        "--save", default=False, action='store_true',
        help="Save animation")
    output_group.add_argument(
        "--cmap", default=None, type=str,
        help="Matplotlib colormap for animation")
    output_group.add_argument(
        "--power_perceptual", default=1.0, type=float, action=CheckArgsPowerPerceptual,
        help="Power perceptual for colormap (default: 1.0, no perceptual scaling)")
    output_group.add_argument(
        "--detail_runhist", default=False, action='store_true',
        help="Compute detailed running history (may take longer time)")
    output_group.add_argument(
        "--show_kkt_it", "--show_kkt", default=False, action='store_true',
        help="Display KKT history vs iterations")
    output_group.add_argument(
        "--save_kkt_it", "--save_kkt", default=False, action='store_true',
        help="Save KKT history vs iterations plot")
    output_group.add_argument(
        "--show_kkt_time", default=False, action='store_true',
        help="Display KKT history vs time")
    output_group.add_argument(
        "--save_kkt_time", default=False, action='store_true',
        help="Save KKT history vs time plot")
    output_group.add_argument(
        "-o", "--outdir", default="output/undated", type=str,
        help="Output directory path")
    output_group.add_argument(
        "--log_file", default=None, type=str,
        help="Log file path")

    # Experimental arguments
    exp_group = parser.add_argument_group(
        "Experimental features")
    exp_group.add_argument(
        "--log_level", type=str, choices=["debug", "kkt", "scaling", "info"], default="info",
        help="Log level: debug=all, kkt=KKT details, scaling=scaling info, info=convergence")
    exp_group.add_argument(
        "--versus_exact", default=False, action='store_true',
        help="Compare with exact transportation (requires setting file support)")
    exp_group.add_argument(
        "--checkpoints", default=None, nargs="+", type=float,
        help="Checkpoints for saving intermediate results (only for `--versus_exact`)")
    exp_group.add_argument(
        "--n_space", type=int, default=None,
        help="Spatial grid points for mesh generation (.py mesh files which require kwarg `n: int` only)")

    if return_parser:
        return parser
    else:
        return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    set_logging_level(log_level=args.log_level, log_file=args.log_file)
    print_example_info(args)

    if not args.versus_exact:
        run_dot_surface(
            solver_name="socp",
            opts=args
        )
    else:
        run_dot_surface_versus_exact(
            solver_name="socp",
            opts=args
        )
