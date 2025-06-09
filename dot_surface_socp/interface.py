"""Core interface module for DOT-Surface-SOCP.

Provides essential functions for problem configuration, solver execution, and result visualization.
Implements standardized interfaces for DOT solvers, geometry normalization, and solution evaluation.

For easy usage and comparison with other algorithms in future,
this module includes some unified tools for running the DOT-Surface algorithm in run_dot_surface() function.
It recieves an optional solver argument, which is a callable function that solves the DOT-Surface problem with standardized inputs and outputs.
This allows for easy integration of different solvers while maintaining a consistent interface.
"""

import numpy as np
from pathlib import Path

# Import the useful routines
from dot_surface_socp.data.load_example import load_example
from dot_surface_socp.utils.evaluate_solution import check_mass_conservation, check_negative_mass
from dot_surface_socp.socp.data_preprocessing import normalize_geometry

from argparse import Namespace as argparse_namespace
import logging
from dot_surface_socp.config import LOG_LEVELS


def print_example_info(opts, fields = None, additional_fields = None):
    """Print the problem info.

    Parameters
    ----------
    opts : argparse.Namespace
        Command line arguments containing problem configuration
    fields : list of str, optional
        List of fields to print, by default None
    additional_fields : list of str, optional
        List of additional fields to print, by default None
    """
    assert isinstance(opts, argparse_namespace), "opts must be an argparse.Namespace object"
    assert fields is None or isinstance(fields, list), "fields must be a list of strings"
    assert additional_fields is None or isinstance(additional_fields, list), "additional_fields must be a list of strings"

    if fields is None:
        fields = [
            'example',
            'mesh_file',
            'setting_file',
            'congestion',
            'ntime',
            'tol',
            'tau',
            'eps',
            'nit',
            'power_perceptual',
        ]

        if hasattr(opts, "save") and opts.save:
            fields.append('outdir')

    if additional_fields:
        fields.extend(additional_fields)

    msg_param = []

    for param in fields:
        if hasattr(opts, param):
            value = getattr(opts, param)
            if value is not None:
                msg_param.append(f'{param}: {value}')

    logging.log(LOG_LEVELS["info"], "") # use as a separator during logging
    logging.log(LOG_LEVELS["info"],
        "---- Info: Experiment Setting ".ljust(42, '-') + "\n"
        f"{'\n'.join(msg_param)}"
    )


def set_logging_level(log_level, log_file = None):
    """Set logging level.

    Parameters
    ----------
    log_level : str
        The logging level to set. Must be one of the keys in config.LOG_LEVELS.
        Available options are typically: 'debug', 'kkt', 'scaling', 'info'.
    log_file : str, optional
        Path to the log file. If provided, logging output will be written to both
        the console and the specified file. If None, output only goes to console.
    """
    _logging_level = LOG_LEVELS.get(log_level, LOG_LEVELS['info'])

    if log_file is not None:
        logging.basicConfig(
            level=_logging_level,
            format='%(asctime)s %(message)s',
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler(log_file)
            ]
        )
    else:
        logging.basicConfig(
            level=_logging_level,
            format='%(message)s'
        )


def run_dot_surface(opts, solver = None, solver_name = None):
    """Main function for solving Dynamic Optimal Transport on discrete surface.

    Parameters
    ----------
    opts : argparse.Namespace
        Command line arguments containing problem configuration
    solver : callable
        A solver function that implements the DOT algorithm interface:
        - Input: (n_time, geometry, **kwargs) where:
            n_time: int - Number of time discretization points
            geometry: dict - Geometric data of the mesh
            **kwargs: Additional solver-specific parameters
        - Output: (solution, run_history) where:
            solution: dict - Contains at least {'mu': ndarray} for density
            run_history: object - Contains optimization history
        Defaults to None, which uses the SOCP solver.
    solver_name : str, optional
        Name of the solver, by default None
    
    Returns
    -------
    solution : dict
        Solution of the DOT problem
    geometry : dict
        Geometric data of the mesh
    run_history : object
        Run history of the solver
    """

    # -----------------------------------------------------------------------------------------------
    # Check Input
    # -----------------------------------------------------------------------------------------------

    # Validate solver interface and expected inputs/outputs
    if solver is None:
        from dot_surface_socp.socp import solver

    if not callable(solver):
        raise TypeError("Solver must be a callable function")
    
    # Validate opts
    if not hasattr(opts, 'ntime') or opts.ntime <= 0:
        raise ValueError("'ntime' must be a positive integer")
    
    if not hasattr(opts, 'example') \
            and not (hasattr(opts, 'mesh_file') \
            and not hasattr(opts, 'setting_file')):
        raise ValueError("Either 'example' or both 'mesh_file' and 'setting_file' must be provided")
    
    if not hasattr(opts, 'show') and not isinstance(opts.show, bool):
        raise ValueError("'show' must be a boolean")

    if not hasattr(opts, 'save') and not isinstance(opts.save, bool):
        raise ValueError("'save' must be a boolean")

    if not hasattr(opts, 'show_kkt_it') and not isinstance(opts.show_kkt_it, bool):
        raise ValueError("'show_kkt_it' must be a boolean")

    if not hasattr(opts,'show_kkt_time') and not isinstance(opts.show_kkt_time, bool):
        raise ValueError("'show_kkt_time' must be a boolean")

    if not hasattr(opts,'save_kkt_it') and not isinstance(opts.save_kkt_it, bool):
        raise ValueError("'save_kkt_it' must be a boolean")

    if not hasattr(opts,'save_kkt_time') and not isinstance(opts.save_kkt_time, bool):
        raise ValueError("'save_kkt_time' must be a boolean")
    
    # Optional parameters validation
    if hasattr(opts, 'outdir') and opts.outdir is not None and not isinstance(opts.outdir, str):
        raise ValueError("'outdir' must be a string")
    
    if hasattr(opts, 'tau') and opts.tau is not None and (opts.tau <= 0 or opts.tau > 2):
        raise ValueError("'tau' must be in range (0, 2]")
    
    if hasattr(opts, 'tol') and opts.tol is not None and opts.tol <= 0:
        raise ValueError("'tol' must be positive")
    
    if hasattr(opts, 'congestion') and opts.congestion is not None and opts.congestion < 0:
        raise ValueError("'congestion' must be non-negative")
    
    if hasattr(opts, 'nit') and opts.nit is not None and opts.nit <= 0:
        raise ValueError("'nit' must be positive")
    
    if hasattr(opts, 'checkpoints') and opts.checkpoints is not None and \
            not isinstance(opts.checkpoints, list) and all(0 < item <= 1 for item in opts.checkpoints):
        raise ValueError("'checkpoints' must be a list of floats in (0, 1]")
    
    if hasattr(opts, 'time_limit') and opts.time_limit is not None and opts.time_limit <= 0:
        raise ValueError("'time_limit' must be positive")
    
    if hasattr(opts, "power_perceptual") and opts.power_perceptual is not None and opts.power_perceptual <= 0:
        raise ValueError("'power_perceptual' must be positive")
    
    if hasattr(opts, "eps") and opts.eps is not None and opts.eps < 0:
        raise ValueError("'eps' must be non-negative")
    
    # -----------------------------------------------------------------------------------------------
    # Preparation
    # -----------------------------------------------------------------------------------------------

    # Load args
    n_time = opts.ntime

    is_show = opts.show
    is_save_animation = opts.save
    is_save_pictures = opts.save

    is_show_kkt_it = opts.show_kkt_it
    is_show_kkt_time = opts.show_kkt_time
    is_save_kkt_it = opts.save_kkt_it
    is_save_kkt_time = opts.save_kkt_time

    # Algorithm name
    algo_name = solver_name if solver_name else solver.__name__

    # Load example and its corresponding view settings
    example_name, geometry, camera_config \
        = load_example(
            example_name=opts.example,
            path_to_mesh_file=opts.mesh_file, path_to_setting_file=opts.setting_file,
            kwargs_generating_mesh={"n": getattr(opts, 'n_space', None)}
        )

    # Format of result files
    outdir = Path(opts.outdir if hasattr(opts, 'outdir') else 'output')

    # Print the problem info
    logging.log(LOG_LEVELS["info"],
        "---- Discretization ".ljust(42, '-') + "\n"
        f"Example name: {example_name}\n"
        f"Number of points in time: {n_time}\n"
        f"Number of vertices: {geometry['vertices'].shape[0]}\n"
        f"Number of triangles: {geometry['triangles'].shape[0]}\n"
        f"Area of the vertices: {np.sum(geometry['area_vertices'] / 3.0)}\n"
        f"Area of the triangles: {np.sum(geometry['area_triangles'])}"
    )

    if is_save_animation or is_save_pictures:
        output_animation_path = Path(outdir / "animation/{subfolder}".format(subfolder=example_name))
        output_animation_path.mkdir(parents=True, exist_ok=True)

        animation_filename = str(output_animation_path / f"{example_name}_{algo_name}.mp4")
        animation_picture_filename = str(output_animation_path / f"{example_name}_{algo_name}_{{time_frame_number}}.png")
        example_picture_filename = str(output_animation_path / f"{example_name}_{{description}}.png")
    else:
        animation_filename, animation_picture_filename, example_picture_filename = None, None, None

    if is_save_kkt_it or is_save_kkt_time:
        output_kkt_path = Path(outdir / "running_history")
        output_kkt_path.mkdir(parents=True, exist_ok=True)

        if is_save_kkt_it:
            kkt_it_filename = str(output_kkt_path / "{example_name}_{algo}_kkt_it.png".format(
                example_name=example_name, algo=algo_name))
        else:
            kkt_it_filename = None

        if is_save_kkt_time:
            kkt_time_filename = str(output_kkt_path / "{example_name}_{algo}_kkt_time.png".format(
                example_name=example_name, algo=algo_name))
        else:
            kkt_time_filename = None
    else:
        kkt_it_filename, kkt_time_filename = None, None

    # -----------------------------------------------------------------------------------------------
    # Call the algorithm
    # -----------------------------------------------------------------------------------------------
    key_mapping = {
        "eps": "eps",
        "tau": "tau",
        "nit": "nit",
        "tol": "tol",
        "congestion": "congestion",
        "checkpoints": "tol_checkpoints",
        "time_limit": "time_limit",
        "detail_runhist": "check_kkt_step_by_step",
    }

    optional_args = {}
    for opts_key, solver_key in key_mapping.items():
        if hasattr(opts, opts_key):
            value = getattr(opts, opts_key)
            if value is not None:
                optional_args[solver_key] = value

    # Call the solver with unified interface
    normalized_geometry, scale_factor = normalize_geometry(geometry)
    solution, run_history = solver(n_time, normalized_geometry, **optional_args)

    # Validate solver output
    if not isinstance(solution, dict) or 'mu' not in solution:
        raise ValueError("Solver must return a solution dictionary containing 'mu' key")

    # Compute functions values on the original geometry
    values_history = run_history.history
    values_keys = ["Transportation cost", "Objective value"]
    area_descale_factor = 1.0 / scale_factor ** 2

    for key in values_keys:
        if key in values_history:
            values_history[key] = area_descale_factor * values_history[key]

    # -----------------------------------------------------------------------------------------------
    # Evaluate the solution
    # -----------------------------------------------------------------------------------------------
    check_mass_conservation(solution["mu"], verbose=True)
    check_negative_mass(solution["mu"], verbose=True)

    # -----------------------------------------------------------------------------------------------
    # Running history
    # -----------------------------------------------------------------------------------------------

    from dot_surface_socp.utils.admm_tools import RunningHistory
    
    if isinstance(run_history, RunningHistory):
        run_history.print_end_history()
        run_history.print_steps_time()

        fig_title = f"{algo_name} solves example '{example_name}'"

        if is_show_kkt_it or is_save_kkt_it:
            run_history.show_kkt_errors(filename=kkt_it_filename, is_show_when_save=is_show_kkt_it, title=fig_title, x_axis='iteration')

        if is_show_kkt_time or is_save_kkt_time:
            run_history.show_kkt_errors(filename=kkt_time_filename, is_show_when_save=is_show_kkt_time, title=fig_title, x_axis='time')
    else:
        # raise ValueError("Solver must return a run_history object")
        pass

    # -----------------------------------------------------------------------------------------------
    # Visualization with PyVista
    # -----------------------------------------------------------------------------------------------

    if not any([is_save_pictures, is_save_animation, is_show]):
        return solution, geometry, run_history
    
    # Import the visualization tools
    from pyvista import set_plot_theme as set_pv_plot_theme
    from dot_surface_socp.utils.show import create_pv_mesh, normalize_density_to_plot, decorator_factory_power_perceptual, \
        save_animation, show_animation, save_description_of_dot, save_results_of_dot

    set_pv_plot_theme("document")
    mesh = create_pv_mesh(geometry["vertices"], geometry["triangles"])
    cmap = opts.cmap if hasattr(opts, "cmap") else None

    # Normalization for better visualization
    power = opts.power_perceptual if hasattr(opts, "power_perceptual") else None
    normalize_to_plot = decorator_factory_power_perceptual(power=power)(normalize_density_to_plot)
    to_plot, to_plot_mu0, to_plot_mu1 = normalize_to_plot(solution["mu"], geometry)

    # Show the animation
    if is_show:
        show_animation(
            mesh, to_plot, example_name, camera_config=camera_config, cmap=cmap
        )
    
    # Save the evolution of densities and the mesh
    if is_save_pictures:
        if example_picture_filename:
            save_description_of_dot(
                mesh, to_plot_mu0, to_plot_mu1, example_picture_filename, camera_config=camera_config, cmap=cmap
            )
        
        if animation_picture_filename:
            num_frames = 7
            save_results_of_dot(
                mesh, to_plot, animation_picture_filename, num_frames=num_frames, camera_config=camera_config, cmap=cmap
            )
    
    # Save the animation 
    if is_save_animation and animation_filename:
        save_animation(
            mesh, to_plot, animation_filename, camera_config=camera_config, cmap=cmap
        )

    return solution, geometry, run_history


def run_dot_surface_versus_exact(opts, solver = None, solver_name = None, type_time_grid : str = "center"):
    """Run DOT-Surface and compare with exact transportation.

    Parameters
    ----------
    opts : argparse.Namespace
        Command line arguments containing problem configuration.
    solver : callable
        A solver function that implements the DOT algorithm interface:
        - Input: (n_time, geometry, **kwargs) where:
            n_time: int - Number of time discretization points
            geometry: dict - Geometric data of the mesh
            **kwargs: Additional solver-specific parameters
        - Output: (solution, run_history) where:
            solution: dict - Contains at least {'mu': ndarray} for density
            run_history: object - Contains optimization history
        Defaults to None, which uses the SOCP solver.
    solver_name : str, optional
        Name of the solver, by default None.
    type_time_grid: str, optional
        Type of time grid, either "staggered" or "center" (default).
    
    Returns
    -------
    solution : dict
        Solution of the DOT problem.
    geometry : dict
        Geometric data of the mesh.
    run_history : object
        Run history of the solver.
    error_transportation : dict (utils.type.ErrorVersusExactData)
        Error versus the exact transportation.
    error_checkpoints : list of dict (utils.type.ErrorVersusExactData)
        Errors of checkpoints versus the exact transportation, if opts.checkpoints is not empty.
    """

    if type_time_grid not in ["center", "staggered"]:
        raise ValueError("type_time_grid must be either 'center' or 'staggered'")
    
    from dot_surface_socp.data.load_example import load_exact_transportation
    from dot_surface_socp.utils.evaluate_solution import compare_with_exact_transportation

    # Compute the exact transportation
    n_time = opts.ntime
    t_array_center = np.linspace(0.0, 1.0, n_time + 1)
    if type_time_grid == "center":
        t_array = t_array_center
    elif type_time_grid == "staggered":
        t_array_staggered = 0.5 * (t_array_center[:-1] + t_array_center[1:])
        t_array = t_array_staggered
    else:
        raise ValueError("Unknown value of type_time_grid")
    
    example_id, exact_transportation = load_exact_transportation(
        t_array=t_array,
        example_name=opts.example,
        path_to_mesh_file=opts.mesh_file, path_to_setting_file=opts.setting_file,
        kwargs_generating_mesh={"n": opts.n_space} if hasattr(opts, "n_space") else None
    )

    # Call the main function to run DOT-Surface
    solution, geometry, run_history = run_dot_surface(opts=opts, solver=solver, solver_name=solver_name)
    transportation = solution["mu"]

    # Evaluate the transportation versus the exact one
    error_transportation = compare_with_exact_transportation(
        mu=transportation, mu_exact=exact_transportation, geometry=geometry, verbose=True
    )

    # Evaluate checkpoints
    error_checkpoints = []
    if "checkpoints" in solution and solution["checkpoints"]:
        for checkpoint in solution["checkpoints"]:
            error_checkpoint = compare_with_exact_transportation(
                mu=checkpoint["mu"], mu_exact=exact_transportation, geometry=geometry, verbose=False
            )
            error_checkpoints.append(
                {
                    "error": error_checkpoint,
                    "kkt_error": max([kkt for kkt in checkpoint["kkt"] if kkt is not None]),
                    "iteration": checkpoint["iteration"],
                    "time": checkpoint["time"]
                }
            )
    
        # Export table
        from dot_surface_socp.utils.file_process import export_table_from_checkpoints_error
        algo_name = solver_name if solver_name else solver.__name__
        out_table = Path(opts.outdir if hasattr(opts, 'outdir') else 'output') / f"error_versus_exact_{algo_name}.html"
        export_table_from_checkpoints_error(
            error_checkpoints=error_checkpoints,
            out=out_table
        )

    return solution, geometry, run_history, error_transportation, error_checkpoints
