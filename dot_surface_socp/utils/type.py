from typing import TypedDict
from numpy import ndarray, newaxis
from typing import TypedDict, NotRequired, Optional


class GeometryData(TypedDict, total=True):
    mu0: ndarray
    mu1: ndarray
    vertices: ndarray
    triangles: ndarray
    edges: ndarray
    area_triangles: NotRequired[ndarray]
    area_vertices: NotRequired[ndarray]

class CheckpointDotData(TypedDict, total=False):
    mu: ndarray
    E: ndarray
    iteration: int
    time: float
    kkt: list[float]

class SolutionSocpData(TypedDict, total=False):
    # Primal variables
    phi: ndarray
    A: ndarray
    B: ndarray
    lambda_c: ndarray
    z_fst: ndarray
    z_mid: ndarray
    z_end: ndarray
    # Dual variables
    mu: ndarray
    E: ndarray
    beta_fst: ndarray
    beta_mid: ndarray
    beta_end: ndarray
    # Checkpoints
    checkpoints: NotRequired[list[CheckpointDotData]]

class SolutionDotData(TypedDict, total=False):
    # Transportation
    mu: ndarray
    E: ndarray
    # Checkpoints
    checkpoints: NotRequired[list[CheckpointDotData]]


def translate_solution_socp_to_dot(solution_socp: SolutionSocpData, geom: GeometryData) -> SolutionDotData:
    solution_dot = SolutionDotData(
        mu=solution_socp["mu"] * (geom["area_vertices"][newaxis, :] / 3.0),
        E=solution_socp["E"] * (geom["area_triangles"][newaxis, :, newaxis])
    )
    
    if "checkpoints" in solution_socp and solution_socp["checkpoints"]:
        solution_dot["checkpoints"] = [
            CheckpointDotData(
                mu=checkpoint["mu"] * (geom["area_vertices"][newaxis, :] / 3.0),
                E=checkpoint["E"] * (geom["area_triangles"][newaxis, :, newaxis]),
                iteration=checkpoint["iteration"],
                time=checkpoint["time"],
                kkt=checkpoint["kkt"]
            ) for checkpoint in solution_socp["checkpoints"]
        ]
    
    return solution_dot


class CameraConfig(TypedDict):
    position: list[float]
    focal_point: list[float]
    up: list[float]
    name: Optional[str] = None

class ErrorVersusExactData(TypedDict):
    l1: float
    l2: float
    linf: float

class CheckpointsErrorData(TypedDict):
    error: ErrorVersusExactData
    kkt_error: float # max(kkt errors)
    iteration: int
    time: float
