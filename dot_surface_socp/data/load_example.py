from typing import Union
import numpy as np
from enum import Enum, auto
from pathlib import Path
import json
import textwrap
import warnings
from importlib.util import spec_from_file_location, module_from_spec

from dot_surface_socp.utils.surface_pre_computations_socp import geometricQuantities, trianglesToVertices
from dot_surface_socp.data.util import read_mesh
from dot_surface_socp.utils.type import GeometryData
import dot_surface_socp.data.settings as settings
from dot_surface_socp.config import PATHS

class TypeLoadingExample(Enum):
    Nan = auto()
    Predefined = auto() # Load the predefined examples, which is located in ./data/ path
    Custom = auto() # Load a custom example, specified by two files: mesh_file.off, setting_file.py


def load_example_file(
        example_name: str = None,
        path_to_mesh_file: str = None, path_to_setting_file: str = None, is_pymesh: bool = False
    ):
    """Load example files

    Parameters
    ----------
    example_name : str, optional
        Name of the predefined example to load, by default None
    path_to_mesh_file : str, optional
        Path to the custom mesh file (.off), by default None
    path_to_setting_file : str, optional
        Path to the custom setting file (.py), by default None
    is_pymesh : bool, optional
        If True, the mesh file is expected to be a Python script
        (if there is a corresponding .py file in ./meshes dir, otherwise using .off file)
        that generates the mesh, by default False

    This function supports two ways of loading examples:
    1. Load predefined examples by providing example_name
    2. Load custom examples by providing paths to mesh file (.off) and setting file (.py)

    Usage
    1. Load predefined example
        >>> example = "airplane"
        >>> load_example_file(example_name=example)
    2. Load custom example
        >>> mesh_file = "./mesh.off"
        >>> setting_file = "./setting.py"
        >>> load_example_file(path_to_mesh_file=mesh_file, path_to_setting_file=setting_file)

    Args:
        example_name: Name of predefined example to load
        path_to_mesh_file: Path to custom mesh file (.off)
        path_to_setting_file: Path to custom setting file (.py)

    Returns:
        tuple: (example_id, mesh_file_name, example_setting)
    """

    if example_name is not None:
        type_predefined = TypeLoadingExample.Predefined
    else:
        type_predefined = TypeLoadingExample.Nan

    if path_to_mesh_file is not None and path_to_setting_file is not None:
        type_custom = TypeLoadingExample.Custom
    else:
        type_custom = TypeLoadingExample.Nan

    type_list = [type_predefined, type_custom]
    idx_chosen_type = [idx for idx, t in enumerate(type_list) if t is not TypeLoadingExample.Nan]

    if len(idx_chosen_type) == 0:
        raise ValueError("The user must choose a way to load example:"
                         "1. Load a predefined example by argument `example_name` "
                         "2. Load a custom example by specifying mesh file (.off) and setting file (.py)")
    elif len(idx_chosen_type) > 1:
        raise ValueError("The user can only load example either by a predefined `example_name` "
                         "or by custom specified mesh file (.off) and setting file (.py)")
    else:
        type_load = type_list[idx_chosen_type[0]]

    # Get mesh and setting
    if type_load is TypeLoadingExample.Predefined:
        mesh_file_name, example_setting = __match_example(example_name, is_pymesh=is_pymesh)
        example_id = example_name
    elif type_load is TypeLoadingExample.Custom:
        check_custom_example(path_to_mesh_file, path_to_setting_file)
        mesh_file_name, example_setting = path_to_mesh_file, load_module_dynamically(path_to_setting_file)
        example_id = f"{Path(path_to_mesh_file).name}-{Path(path_to_setting_file).name}".replace(".", "_")
    else:
        raise ValueError(f"Type of loading example {type_load} should be either Predefined or Custom")
    
    return example_id, mesh_file_name, example_setting


def load_example(
        example_name: str = None,
        path_to_mesh_file: str = None, path_to_setting_file: str = None,
        kwargs_generating_mesh: dict = None
    ):
    """Load example

    This function supports two ways of loading examples (see load_example_file() for details):
        1. Load predefined examples by providing example_name
        2. Load custom examples by providing paths to mesh file (.off) and setting file (.py)
    """

    # Load example file
    example_id, mesh_file_name, example_setting = load_example_file(
        example_name, path_to_mesh_file, path_to_setting_file, is_pymesh=True if isinstance(kwargs_generating_mesh["n"], int) else False
    )

    # Geometry
    vertices, triangles, edges = read_mesh(
        mesh_file_name,
        kwargs_generating_mesh=kwargs_generating_mesh
    )
    area_triangles, _, _ = geometricQuantities(vertices, triangles, edges)
    _, area_vertices, _, _ = trianglesToVertices(vertices, triangles, area_triangles)
    mub0, mub1 = example_setting.get_mu(area_vertices, vertices)

    # Camera parameters
    try:
        camera_file = Path(PATHS["camera_dir"]) / f"{example_id}.json"
        if camera_file.exists():
            with open(camera_file, 'r') as f:
                camera_config = json.load(f)
        else:
            camera_config = None
    except Exception as e:
        raise ValueError(f"Failed to load camera parameters with exception: {e}")

    # Normalization
    mub0 /= np.sum(mub0)
    mub1 /= np.sum(mub1)

    geometry = GeometryData(
        vertices=vertices,
        triangles=triangles,
        edges=edges,
        mu0=mub0,
        mu1=mub1,
        area_triangles=area_triangles,
        area_vertices=area_vertices
    )

    return example_id, geometry, camera_config

def load_exact_transportation(
        t_array: np.ndarray,
        example_name: str = None,
        path_to_mesh_file: str = None, path_to_setting_file: str = None,
        kwargs_generating_mesh: dict = None
    ):
    """Load exact transportation

    This function supports two ways of loading exact transportation (see load_example_file() for details):
        1. Load predefined one by providing example_name
        2. Load custom one by providing paths to mesh file (.off) and setting file (.py)
    
    Returns:
        example_id: str, example name
        exact_transportation: np.ndarray, exact transportation for each time slot
    """

    # Load example file
    example_id, mesh_file_name, example_setting = load_example_file(
        example_name, path_to_mesh_file, path_to_setting_file, is_pymesh=True if isinstance(kwargs_generating_mesh["n"], int) else False
    )

    # Geometry
    vertices, triangles, edges = read_mesh(
        mesh_file_name,
        kwargs_generating_mesh=kwargs_generating_mesh
    )
    area_triangles, _, _ = geometricQuantities(vertices, triangles, edges)
    _, area_vertices, _, _ = trianglesToVertices(vertices, triangles, area_triangles)

    # Compute exact transportation
    if hasattr(example_setting, 'get_exact_transportation'):
        exact_transportation = example_setting.get_exact_transportation(
            t_array, vertices, area_vertices
        )
    else:
        raise ValueError(f"The setting file must have defined the function of exact transportation: get_exact_transportation(...).")

    # Normalization
    scale_mu0 = exact_transportation[0].sum()
    scale_mu1 = exact_transportation[-1].sum()
    scale = 0.5 * (scale_mu0 + scale_mu1)
    exact_transportation /= scale

    if abs(scale_mu0 - scale_mu1) > 1e-4:
        warnings.warn(f"There is a big difference between scale of mu0 ({scale_mu0}) and scale of mu1 ({scale_mu1})")

    return example_id, exact_transportation

def load_module_dynamically(path_to_module: Union[Path, str]):
    if isinstance(path_to_module, str):
        path_to_module = Path(path_to_module)

    module_name = path_to_module.stem
    spec = spec_from_file_location(module_name, str(path_to_module))
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

def check_custom_example(mesh_filename: Union[Path, str], setting_filename: Union[Path, str]):
    if isinstance(mesh_filename, str):
        mesh_filename = Path(mesh_filename)

    if mesh_filename.suffix != ".off":
        raise FileExistsError("The input mesh file should be .off file")

    if isinstance(setting_filename, str):
        setting_filename = Path(setting_filename)

    setting = load_module_dynamically(setting_filename)
    if not all([hasattr(setting, "get_mu")]):
        raise FileExistsError("The input setting file should have defined the function:"
                              "\tget_mu(area_vertices, vertices)")

def __match_example(name: str, is_pymesh: bool = False):
    # Get path configuration from config module
    path_meshes = Path(PATHS["mesh_dir"])
    path_settings = Path(PATHS["setting_dir"])
    invalid_input = [
        file.name[:-3] for file in path_settings.glob('*.py') if file.name not in ["default.py", "__init__.py"]
    ]

    match name:
        # ==== examples from lavenant-2021 (https://doi.org/10.1145/3272127.3275064)
        case "airplane":
            file_name = path_meshes / "airplane_62.off"
            setting = settings.airplane
        case "armadillo":
            file_name = path_meshes / "armadillo.off"
            setting = settings.armadillo
        case "square_regular":
            file_name = path_meshes / "square_regular_100.off"
            setting = settings.square_regular
        case "face":
            file_name = path_meshes / "face_vector_field_319.off"
            setting = settings.face
        case "hand":
            file_name = path_meshes / "hand_3k.off"
            setting = settings.hand
        case "sphere":
            file_name = path_meshes / "sphere_puncture.off"
            setting = settings.sphere
        # ==== Newly defined examples
        case "punctured_ball":
            file_name = path_meshes / "punctured_ball.off"
            setting = settings.punctured_ball
        case "bunny":
            file_name = path_meshes / "bunny.off"
            setting = settings.bunny
        case "ring":
            file_name = path_meshes / "ring.off"
            setting = settings.ring
        case "eight":
            file_name = path_meshes / "eight.off"
            setting = settings.eight
        case "audi":
            file_name = path_meshes / "audi.off"
            setting = settings.audi
        case "knots_3":
            file_name = path_meshes / "knots_3.off"
            setting = settings.knots_3
        case "knots_5":
            file_name = path_meshes / "knots_5.off"
            setting = settings.knots_5
        case "hills":
            file_name = path_meshes / "hills.off"
            setting = settings.hills
        case "robot":
            file_name = path_meshes / "robot.off"
            setting = settings.robot
        case "plane":
            if is_pymesh:
                file_name = path_meshes / "plane.py"
            else:
                file_name = path_meshes / "plane.off"
            setting = settings.plane
        # ==== Refined examples
        case "refined_airplane":
            file_name = path_meshes / "refined_airplane_62.off"
            setting = settings.refined_airplane
        case "refined_armadillo":
            file_name = path_meshes / "refined_armadillo.off"
            setting = settings.refined_armadillo
        case "refined_face":
            file_name = path_meshes / "refined_face_vector_field_319.off"
            setting = settings.refined_face
        case "refined_hand":
            file_name = path_meshes / "refined_hand_3k.off"
            setting = settings.refined_hand
        case "refined_punctured_ball":
            file_name = path_meshes / "refined_punctured_ball.off"
            setting = settings.refined_punctured_ball
        case "refined_bunny":
            file_name = path_meshes / "refined_bunny.off"
            setting = settings.refined_bunny
        case x if x in invalid_input:
            # Failed to statically load example, try to dynamically load
            try:
                file_name = path_meshes / f"{x}.off"
                setting = getattr(settings, x)
                warnings.warn(f"Had dynamically loaded the example '{x}', as failed to load it statically.\n"
                              f"Specifically, the input '{x}' had matched a setting file (./data/settings/{x}.py), "
                              f"but it was not correctly matched in ./data/load_example.py.")
            except Exception as e:
                raise ValueError(f"Failed to dynamically load the example '{x}' with exception: {e}\n")
        case _:
            msg = textwrap.wrap(f"[{', '.join(invalid_input)}]", width=100)
            raise ValueError("The chosen example is not valid. Invalid values:\n" + "\n".join(msg))

    return file_name, setting