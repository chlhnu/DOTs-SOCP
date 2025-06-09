"""SOCP for Dynamical Optimal Transport on Surfaces.
"""

from functools import partial
from typing import Union
import numpy as np
import numexpr as ne
import threading
import scipy.sparse as scsp
from scipy.ndimage import correlate1d
from math import sqrt
import time

from dot_surface_socp.utils import admm_tools
from dot_surface_socp.utils.surface_pre_computations_socp import geometricQuantities, geometricMatrices, trianglesToVertices
from dot_surface_socp.utils.laplacian_inverse_socp import buildLaplacianMatrix
from dot_surface_socp.utils.type import GeometryData, SolutionSocpData, CheckpointDotData
from dot_surface_socp.utils.condition_validator import create_convergence_condition_validator, ConvergenceConditionManager, max_of_list_with_none
from dot_surface_socp.utils.condition_validator_wrapper import AdaptiveValidatorWrapper

import logging
from dot_surface_socp.config import LOG_LEVELS


def solver_socp(
        n_time,
        geometry: GeometryData,
        congestion=0.0,
        nit=1000,
        eps=0.0*10**(-8),
        tol=1e-4,
        tau=1.90,
        is_palm=False,
        is_multi_threads=True,
        is_z_scaling=True,
        is_constant_scaling=False,
        check_kkt_step_by_step=False,
        init_solution=SolutionSocpData(),
        tol_checkpoints=None,
        time_limit=1000,
    ):
    """SOCP for Dynamical Optimal Transport on Discrete Surfaces.

    Parameters
    ----------
    n_time : int
        Number of discretization points in time
    geometry : GeometryData
        The geometric quantities of the mesh
    congestion : float, optional
        The congestion parameter, by default 0.0
    nit : int, optional
        Maximum number of iterations, by default 1000
    eps : float, optional
        The regularization parameter of the Laplacian operator, by default 0.0*10**(-8)
    tol : float, optional
        Tolerance of the solution, by default 1e-4
    tau : float, optional
        Step-size of the iALM or ALM, by default 1.90
    is_palm : bool, optional
        Whether to use ALM, by default False
    is_multi_threads : bool, optional
        Whether to use multi-threads, by default True
    is_z_scaling : bool, optional
        Whether to use scaling, by default True
    is_constant_scaling : bool, optional
        Whether to use constant scaling, by default False
    check_kkt_step_by_step : bool, optional
        Whether to check all KKT conditions step by step, by default False
    init_solution : dict, optional
        Initial solution for warm start, by default None
    tol_checkpoints : list, optional
        Tolerance of the solution at each checkpoint, by default None
    time_limit : float, optional
        Time limit for the solver, by default 1000

    Returns
    -------
    tuple (SolutionSocpData, admm_tools.RunningHistory)
    """
    # The default setting of logging
    logging.basicConfig(level=LOG_LEVELS["info"], format='%(message)s')

    # Validate tol_checkpoints if provided
    checkpoint_solutions = []
    if tol_checkpoints is not None:
        if not isinstance(tol_checkpoints, list) or not tol_checkpoints:
            raise ValueError("tol_checkpoints must be a non-empty list")
        for i, checkpoint in enumerate(tol_checkpoints):
            if not (isinstance(checkpoint, (int, float)) and 0 < checkpoint < 1):
                raise ValueError(f"Invalid checkpoint value at index {i}: {checkpoint}. Must be between 0 and 1")
            if checkpoint < tol:
                raise ValueError(f"Checkpoint value must be greater than tol. However, checkpoint ({checkpoint}) < tol ({tol})")
        tol_checkpoints = sorted(tol_checkpoints, reverse=True)

    # Parameters
    r = 1.0
    
    # Discretization
    stepsize_time = 1.0 / n_time

    vertices, triangles, edges = geometry["vertices"], geometry["triangles"], geometry["edges"]
    area_triangles, angle_triangles, base_function = geometricQuantities(
        vertices, triangles, edges
    )
    mat_grad_space, mat_div_space, mat_lap_space = geometricMatrices(
        vertices, triangles, edges, area_triangles, angle_triangles, base_function
    )
    _map_triangle_to_vertice, area_vertices, _map_vertice_to_triangle, area_vertices_triangles = trianglesToVertices(
        vertices, triangles, area_triangles
    )
    area_vertices = area_vertices / 3.0
    area_vertices_triangles = area_vertices_triangles / 3.0

    # Size
    n_vertices = vertices.shape[0]
    n_triangles = triangles.shape[0]
    
    # Print the problem info
    logging.log(LOG_LEVELS["kkt"], 
        "---- Experiment info ".ljust(42, '-') + "\n"
        f"Congestion parameter: {congestion}"
        f"Number of discretization points in time: {n_time}\n"
        f"Number of discretization vertices: {n_vertices}\n"
        f"Number of discretization triangles: {n_triangles}\n"
        f"Stepsize: {tau}\n"
        f"Is multiple threads: {is_multi_threads}"
    )

    if is_multi_threads:
        ne.set_num_threads(max(1, min(4, ne.detect_number_of_cores() // 2)))
    else:
        ne.set_num_threads(max(1, min(8, ne.detect_number_of_cores())))

    # Total area
    area_mesh = np.sum(area_triangles)

    # Vectorized arrays
    area_triangle_space = np.kron(
        np.kron(np.ones(n_time + 1), area_triangles),
        np.ones(3)
    ).reshape(n_time + 1, n_triangles, 3)
    
    area_triangle_space_decouple = np.kron(
        np.kron(np.ones(6 * n_time), area_triangles),
        np.ones(3)
    ).reshape(n_time, 2, 3, n_triangles, 3)
    
    area_vertice_time = np.kron(
        np.ones(n_time),
        area_vertices
    ).reshape((n_time, n_vertices))
    
    area_vertices_center = np.kron(
        np.ones(n_time + 1),
        area_vertices
    ).reshape((n_time + 1, n_vertices))

    # Vectorized matrices
    #   Map: V to 3T with 1 weight
    map_vertice_to_triangle = scsp.kron(scsp.eye(n_time), _map_vertice_to_triangle).tocsr()
    #   Map: V to T with 1/3 weight
    map_decouple_space = scsp.kron(
        (1.0 / 3.0) * scsp.eye(n_time + 1),
        _map_vertice_to_triangle[:n_triangles, :] + _map_vertice_to_triangle[n_triangles:2*n_triangles, :] + _map_vertice_to_triangle[2*n_triangles:, :]
    ).tocsr()
    #   Map: 3T to V with weight |f_k|
    map_triangle_to_vertice = scsp.kron(scsp.eye(n_time), _map_triangle_to_vertice).tocsr()
    #   Map: 3T to V with 1 weight
    map_triangle_to_vertice_one = scsp.kron(scsp.eye(n_time), _map_vertice_to_triangle.transpose()).tocsr()
    #   Map: z_mid to x_mid = sqrt(|f_k| / (|v|)) z_mid for each k in T_v
    diagonal_to_soc = np.kron(
        np.ones(n_time),
        np.sqrt(
            np.divide(
                np.kron(np.ones(3), area_triangles),
                area_vertices_triangles
            )
        ),
    ).reshape(n_time, 3, n_triangles)
    diagonal_to_soc_global = np.kron(
        np.kron(
            np.ones(2 * n_time),
            np.sqrt(
                np.divide(
                    np.kron(np.ones(3), area_triangles),
                    area_vertices_triangles
                )
            )
        ),
        np.ones(3)
    ).reshape(n_time, 2, 3, n_triangles, 3)
    #   Diagonal element of matrix in step q
    def assemble_diagonal_b(scale_z: float = 1.0):
        _diagonal_b = 1.0 + (2.0 * scale_z ** 2) * np.ones(n_time + 1)
        _diagonal_b[0] = 1.0 + scale_z ** 2
        _diagonal_b[-1] = 1.0 + scale_z ** 2
        _diagonal_b = np.kron(
            _diagonal_b,
            np.ones(n_triangles * 3)
        ).reshape(n_time + 1, n_triangles, 3)
        return _diagonal_b

    # Build the Laplacian matrix
    laplacian_invert = buildLaplacianMatrix(
        n_time=n_time,
        stepsize_time=stepsize_time,
        n_vertices=n_vertices,
        area_vertices=area_vertices,
        laplacian_space=mat_lap_space,
        eps=eps,
    )

    # Tools
    norm_square_center = partial(norm_square_weight, area_vertices_center, n_time + 1)
    norm_square_time = partial(norm_square_weight, area_vertice_time, n_time)
    norm_square_space = partial(norm_square_weight, area_triangle_space, n_time + 1)
    norm_square_space_decouple = partial(norm_square_weight, area_triangle_space_decouple, n_time)

    grad_time = partial(vanilla_grad_time, stepsize_time)
    div_time = partial(vanilla_div_time, stepsize_time)
    grad_space = partial(vanilla_grad_space, n_time, n_triangles, mat_grad_space)
    div_space = partial(vanilla_div_space, n_time, n_vertices, n_triangles, mat_div_space)

    solve_laplacian = partial(
        vanilla_solve_laplacian,
        laplacian_invert, div_time, div_space, area_vertice_time, area_triangle_space, area_vertices_center, eps
    )
    solve_proj_soc = partial(
        vanilla_solve_proj_soc,
        diagonal_to_soc_global, diagonal_to_soc, map_triangle_to_vertice_one, map_vertice_to_triangle
    )
    solve_q_lambda = partial(
        vanilla_solve_q_lambda,
        1.0, assemble_diagonal_b()
    )

    # Variable initialization
    phi         = init_solution.get('phi', np.zeros((n_time + 1, n_vertices)))
    A           = init_solution.get('A', grad_time(phi))
    B           = init_solution.get('B', grad_space(phi))
    lambda_c    = init_solution.get('lambda_c', np.zeros((n_time, n_vertices)))
    z_fst       = init_solution.get('z_fst', np.zeros((n_time, n_vertices)))
    z_end       = init_solution.get('z_end', np.zeros((n_time, n_vertices)))
    z_mid       = init_solution.get('z_mid', np.zeros((n_time, 2, 3, n_triangles, 3)))
    beta_fst    = (1.0 / r) * init_solution.get('beta_fst', np.zeros((n_time, n_vertices)))
    beta_end    = (1.0 / r) * init_solution.get('beta_end', np.zeros((n_time, n_vertices)))
    beta_mid    = (1.0 / r) * init_solution.get('beta_mid', np.zeros((n_time, 2, 3, n_triangles, 3)))
    mu          = (1.0 / r) * init_solution.get('mu', r * (beta_fst - beta_end))
    E           = (1.0 / r) * init_solution.get('E', - r * decouple_adjoin_spacial(beta_mid))
    
    #   Gradient of phi
    if is_palm:
        dt_phi = grad_time(phi)
        dx_phi = grad_space(phi)
    else:
        dt_phi, dx_phi = np.array(0.0), np.array(0.0)

    memo_A      = np.zeros((n_time, n_vertices))
    memo_B      = np.zeros((n_time + 1, n_triangles, 3))
    memo_z_fst  = np.zeros((n_time, n_vertices))
    memo_z_mid  = np.zeros((n_time, 2, 3, n_triangles, 3))
    memo_z_mid2 = np.zeros((n_time, 2, 3, n_triangles, 3))
    memo_z_end  = np.zeros((n_time, n_vertices))

    # Build the boundary term
    mub0, mub1 = geometry["mu0"], geometry["mu1"]
    boundary_time_with_area = np.zeros((n_time + 1, n_vertices))
    boundary_time_with_area[0, :]  = - mub0 / (r * stepsize_time)
    boundary_time_with_area[-1, :] = mub1 / (r * stepsize_time)

    # Preparation for running history
    run_history = admm_tools.RunningHistory(
        max_record_numbers=nit,
        kkt_labels=[
            "SOC & Org : Primal Feasibility (q)",
            "SOC       : Primal Feasibility (z)",
            "SOC & Org : Dual Feasibility (alpha)",
            "SOC       : Dual Feasibility (beta)",
            "      Org : ||rho - Pi+(rho + Fq)||",
            "      Org : ||m - rho o B||",
            "      Org : ||cong. rho - lambda_c||",
        ],
        kkt_short_labels=[
            "Prim(phi, q)",
            "Prim(q, z)",
            "Dual(alpha)",
            "Dual(beta)",
            "Comp(rho, f(q))",
            "Comp(m, rho o B)",
            "Comp(rho, cong.)",
        ],
        name="SOCP",
    )
    adjust_params = admm_tools.AdjustAdmmParam()
    norm_boundary = r * stepsize_time * sqrt(norm_square_center(np.divide(boundary_time_with_area, area_vertices_center)))
    norm_constant_d = sqrt(2 * area_mesh) # equivalent with `norm_constant_d = sqrt(2 * norm_square_time(np.ones_like(area_vertice_time)))`

    kkt_stop_condition = [0, 2, 4, 5]
    kkt_prim_pos = [0, 1]
    kkt_dual_pos = [2, 3]

    mean_area_center = np.mean(area_vertices_center)
    mean_area_time = np.mean(area_vertice_time)
    mean_area_space = np.mean(area_triangle_space)
    mean_area_space_decouple = np.mean(area_triangle_space_decouple)

    kkt_const_prim_q = np.mean([mean_area_time, mean_area_space])
    kkt_const_prim_z = np.mean([mean_area_time, mean_area_space_decouple, mean_area_time])
    kkt_const_dual_alpha = mean_area_center
    kkt_const_dual_beta = np.mean([mean_area_time, mean_area_space])
    kkt_const_comp_rho = mean_area_time
    kkt_const_comp_m = mean_area_space

    # ===================================
    # Tools to adjust params and scale
    # ===================================
    prim_scale = 1.0
    dual_scale = 1.0
    constant_d = 1.0
    scale_factor_z = 1.0
    is_org_kkt = False

    def scale_prim_dual(scale_factor: tuple[float, float] = None, msg: str = "Norm of prim and dual"):
        """Rescale the primal and dual variables"""
        nonlocal r, prim_scale, dual_scale, congestion, constant_d, norm_constant_d, norm_boundary

        if scale_factor is None:
            _prim_rescale, _dual_rescale = adjust_params.compute_scale_factor(
                prim_norm=np.array([
                    sqrt(norm_square_time(dt_phi) + norm_square_space(dx_phi)), sqrt(norm_square_time(A) + norm_square_space(B)),
                    sqrt(norm_square_time(z_fst) + norm_square_space_decouple(z_mid) + norm_square_time(z_end))
                ]),
                dual_norm=np.array([
                    r * sqrt(norm_square_time(mu) + norm_square_space(E)),
                    r * sqrt(norm_square_time(beta_fst) + norm_square_space_decouple(beta_mid) + norm_square_time(beta_end))
                ]),
                msg=msg
            )
        else:
            _prim_rescale, _dual_rescale = scale_factor

        # Skip rescaling if the scale factors is too small
        if max(_prim_rescale, _dual_rescale) / min(_prim_rescale, _dual_rescale) > 2.0:
            logging.log(LOG_LEVELS["scaling"], f"Scale/Rescale with (prim, dual) factor: {1.0 / _prim_rescale}, {1.0 / _dual_rescale}")

            # compute scale factors
            prim_scale *= _prim_rescale
            dual_scale *= _dual_rescale

            # Scale primal variables
            for var in [phi, A, B, lambda_c, dt_phi, dx_phi, z_fst, z_mid, z_end]:
                ne.evaluate("var / _prim_rescale", out=var)

            # Scale dual variables
            _dual_factor = _dual_rescale ** 2 / _prim_rescale
            for var in [boundary_time_with_area, mu, E, beta_fst, beta_mid, beta_end]:
                ne.evaluate("var / _dual_factor", out=var)

            # Scale the constant values
            r *= _dual_rescale / _prim_rescale
            congestion *= _dual_rescale / _prim_rescale
            constant_d /= _prim_rescale
            norm_constant_d /= _prim_rescale
            norm_boundary /= _dual_rescale

    def adjust_penalty(factor: float):
        nonlocal r
        r = r * factor
        for var in [mu, E, boundary_time_with_area, beta_fst, beta_mid, beta_end]:
            ne.evaluate("var / factor", out=var)

    def scale_variable_z(scale_factor: float, msg: str = "Scale z"):
        nonlocal scale_factor_z, constant_d, norm_constant_d

        logging.log(LOG_LEVELS["scaling"], f"{msg} with z factor: {scale_factor}")

        scale_factor_z *= scale_factor
        constant_d *= scale_factor
        norm_constant_d *= scale_factor

        # Scale primal and dual variables
        [prim_var.__imul__(scale_factor_z) for prim_var in [z_fst, z_mid, z_end]]
        [dual_var.__imul__(1.0 / scale_factor_z) for dual_var in [beta_fst, beta_mid, beta_end]]

        nonlocal mu, E
        mu = scale_factor_z * (beta_fst - beta_end)
        E = - decouple_adjoin_spacial(beta_mid, scale_z=scale_factor_z)

        # Reassemble the linear system for q and lambda
        nonlocal solve_q_lambda
        solve_q_lambda = partial(
            vanilla_solve_q_lambda,
            scale_factor_z, assemble_diagonal_b(scale_z=scale_factor_z)
        )
    
    def recorver_scaled_solution(checkpoint=False, kwargs={}):
        """Recorver original solution from solution to scaled problem
        """
        nonlocal r, prim_scale, dual_scale, scale_factor_z
        if not checkpoint:
            [prim_var.__imul__(prim_scale) for prim_var in [phi, A, B, lambda_c]]
            [prim_var.__imul__(prim_scale / scale_factor_z) for prim_var in [z_fst, z_mid, z_end]]
            [dual_var.__imul__(r * dual_scale) for dual_var in [mu, E]]
            [dual_var.__imul__(r * scale_factor_z * dual_scale) for dual_var in [beta_fst, beta_mid, beta_end]]
        else:
            _solution = CheckpointDotData(
                mu=(r * dual_scale) * mu,
                E=(r * dual_scale) * E,
                **kwargs
            )
            return _solution
    
    # ===================================
    # Tools to evaluate the solution
    # ===================================
    def objective_functional(_phi, _lambda_c, boundary, _congestion):
        """Computation of the transportation cost and objective functional
        """

        # Transportation cost
        trans_cost = stepsize_time * (np.dot(_phi[0, :], boundary[0, :]) + np.dot(_phi[-1, :], boundary[-1, :]))

        # Congestion
        if _congestion > 10 ** (-10):
            congestion_penalty = 1. / (2. * _congestion) * norm_square_time(_lambda_c)
            lagrangian = trans_cost - congestion_penalty
        else:
            lagrangian = trans_cost

        return trans_cost, lagrangian

    def kkt_primal_q(resi_mu, resi_e, _dt_phi, _dx_phi, q_a, q_b, _lambda_c, _prim_scale: Union[float, list[float]] = 1.0):
        """Error of primal constraints (phi, q)
        """
        norm_sum = (
            sqrt(norm_square_time(_dt_phi) + norm_square_space(_dx_phi))
            + sqrt(norm_square_time(q_a) + norm_square_space(q_b))
            + sqrt(norm_square_time(_lambda_c))
        )

        prim_resi = sqrt(
            norm_square_time(resi_mu)
            + norm_square_space(resi_e)
        )

        if isinstance(_prim_scale, (int, float)):
            return prim_resi / (kkt_const_prim_q / _prim_scale + norm_sum)
        else:
            return [prim_resi / (kkt_const_prim_q / scale + norm_sum) for scale in _prim_scale]

    def kkt_primal_z(resi_beta_fst, resi_beta_mid, resi_beta_end, _prim_scale: Union[float, list[float]] = 1.0):
        """Error of primal constraint (q, z)
        """
        prim_resi = sqrt(
            norm_square_time(resi_beta_fst)
            + norm_square_time(resi_beta_end)
            + norm_square_space_decouple(resi_beta_mid)
        )

        if isinstance(_prim_scale, (int, float)):
            return prim_resi / (kkt_const_prim_z / _prim_scale + norm_constant_d)
        else:
            return [prim_resi / (kkt_const_prim_z / scale + norm_constant_d) for scale in _prim_scale]

    def kkt_dual_alpha(_mu, e, boundary_with_area, penalty, _dual_scale: Union[float, list[float]] = 1.0):
        """Error of dual constraint (mu, e)
        """
        dual_aux = (penalty * stepsize_time) * np.divide(
            (
                boundary_with_area
                + div_time(np.multiply(_mu, area_vertice_time))
                + div_space(np.multiply(e, area_triangle_space))
            ),
            area_vertices_center
        )
        dual_resi = sqrt(norm_square_center(dual_aux))

        if isinstance(_dual_scale, (int, float)):
            return dual_resi / (kkt_const_dual_alpha / _dual_scale + norm_boundary)
        else:
            return [dual_resi / (kkt_const_dual_alpha / scale + norm_boundary) for scale in _dual_scale]

    def kkt_dual_beta(_mu, e, _beta_fst, _beta_mid, _beta_end, penalty, _dual_scale: Union[float, list[float]] = 1.0):
        """Error of dual constraint (alpha, beta)
        """
        aux_beta_1 = scale_factor_z * (_beta_end - _beta_fst)
        aux_beta_2 = decouple_adjoin_spacial(_beta_mid, scale_z=scale_factor_z)

        norm_sum = penalty * (
            sqrt(norm_square_time(_mu) + norm_square_space(e))
            + sqrt(norm_square_time(aux_beta_1) + norm_square_space(aux_beta_2))
        )

        resi = penalty * sqrt(
            norm_square_time(_mu + aux_beta_1)
            + norm_square_space(e + aux_beta_2)
        )

        if isinstance(_dual_scale, (int, float)):
            return resi / (kkt_const_dual_beta / _dual_scale + norm_sum)
        else:
            return [resi / (kkt_const_dual_beta / scale + norm_sum) for scale in _dual_scale]

    def kkt_complement_rho_fq(_mu, q_a, q_b):
        """Error of complementary condition (rho, f(q))
        """
        resi_aux = (
            q_a
            + .25 * np.divide(
                map_triangle_to_vertice.dot(
                    np.sum(np.square(decouple_spacial(q_b)), axis=(1, 4)).reshape(-1)
                ).reshape((n_time, n_vertices)),
                area_vertice_time
            )
        )

        norm_sum = (
            sqrt(norm_square_time(_mu))
            + sqrt(norm_square_time(resi_aux))
        )

        resi_aux = np.maximum(0., resi_aux + _mu) - _mu
        resi = sqrt(norm_square_time(resi_aux))

        return resi / (kkt_const_comp_rho + norm_sum)

    def kkt_complement_m_rho_b(m, rho, b):
        """Error of DOT complementary condition (m, rho, b)
        """
        aux = np.broadcast_to(
            map_decouple_space.dot(
                decouple_adjoint_time(rho).reshape(-1)
            ).reshape((n_time + 1, n_triangles, 1)),
            (n_time + 1, n_triangles, 3)
        )
        aux = np.multiply(aux, b)

        norm_sum = (
            sqrt(norm_square_space(m))
            + sqrt(norm_square_space(aux))
        )

        aux -= m
        resi = sqrt(norm_square_space(aux))

        return resi / (kkt_const_comp_m + norm_sum)
    
    def kkt_complement_congestion(_mu, _lambda_c):
        """Error of congestion complementary condition (mu, lambda_c)
        """
        norm_sum = (
            sqrt(norm_square_time(_mu))
            + sqrt(norm_square_time(_lambda_c))
        )

        resi = sqrt(norm_square_time(congestion * _mu - _lambda_c))

        return resi / (kkt_const_comp_rho + norm_sum)


    # ===================================
    # Main computation
    # ===================================
    run_history.start()
    run_history.create_tol_progress(target_tol=tol)
    counter_main = -1
    prim_gap = 1.0 + 1.0 * np.exp(-100 * congestion)

    # Initial scaling
    if is_z_scaling:
        scale_variable_z(scale_factor=2.0, msg="Initially scale z")

    if is_constant_scaling:
        _boundary_time = r * np.divide(boundary_time_with_area, area_vertices_center)
        _norm_c = sqrt(norm_square_center(_boundary_time))
        _norm_ac = sqrt(norm_square_time(grad_time(_boundary_time)) + norm_square_space(grad_space(_boundary_time)))
        _dual_init_scale = sqrt(n_time) * _norm_c ** 2 / _norm_ac

        _prim_init_scale = norm_constant_d

        scale_prim_dual(
            scale_factor=(_prim_init_scale, _dual_init_scale),
            msg="Var Norm at initial scaling"
        )
        adjust_penalty(1.0 / r)
    
    # KKT validator
    kkt_validator, kkt_collector = create_convergence_condition_validator(
        [
            ("Prim(phi, q)", lambda: kkt_primal_q(
                    ne.evaluate("dt_phi - A - lambda_c", local_dict={"dt_phi": dt_phi, "A": A, "lambda_c": lambda_c}),
                    ne.evaluate("dx_phi - B", local_dict={"dx_phi": dx_phi, "B": B}),
                    dt_phi, dx_phi, A, B, lambda_c, _prim_scale=[prim_scale, 1.0]
                )
            ),
            ("Prim(q, z)", lambda: kkt_primal_z(
                    ne.evaluate("z_fst + scale_factor_z * A - constant_d", local_dict={"z_fst": z_fst, "A": A, "constant_d": constant_d, "scale_factor_z": scale_factor_z}),
                    ne.evaluate("scale_factor_z * (z_mid - memo_z_mid)", local_dict={"z_mid": z_mid, "memo_z_mid": memo_z_mid, "scale_factor_z": scale_factor_z}),
                    ne.evaluate("z_end - scale_factor_z * A - constant_d", local_dict={"z_end": z_end, "A": A, "constant_d": constant_d, "scale_factor_z": scale_factor_z}),
                    _prim_scale=[prim_scale, 1.0]
                )
            ),
            ("Dual(alpha)", lambda: kkt_dual_alpha(
                    mu, E, boundary_time_with_area, r,
                    _dual_scale=[dual_scale, 1.0]
                )
            ),
            ("Dual(beta)", lambda: kkt_dual_beta(
                    mu, E, beta_fst, beta_mid, beta_end, r,
                    _dual_scale=[dual_scale, 1.0]
                )
            ),
            ("Comp(rho, f(q))", lambda: [
                    kkt_complement_rho_fq(
                        ne.evaluate("(dual_scale * r) * mu", local_dict={"dual_scale": dual_scale, "r": r, "mu": mu}),
                        ne.evaluate("prim_scale * A", local_dict={"prim_scale": prim_scale, "A": A}),
                        ne.evaluate("prim_scale * B", local_dict={"prim_scale": prim_scale, "B": B})
                    ),
                    None
                ]
            ),
            ("Comp(m, rho o B)", lambda: [
                    kkt_complement_m_rho_b(
                        ne.evaluate("(dual_scale * r) * E", local_dict={"dual_scale": dual_scale, "r": r, "E": E}),
                        ne.evaluate("(dual_scale * r) * mu", local_dict={"dual_scale": dual_scale, "r": r, "mu": mu}),
                        ne.evaluate("prim_scale * B", local_dict={"prim_scale": prim_scale, "B": B})
                    ),
                    None
                ]
            ),
            ("Comp(rho, cong.)", lambda: [
                    kkt_complement_congestion(
                        ne.evaluate("(dual_scale * r) * mu", local_dict={"dual_scale": dual_scale, "r": r, "mu": mu}),
                        ne.evaluate("prim_scale * lambda_c", local_dict={"prim_scale": prim_scale, "lambda_c": lambda_c})
                    ),
                    None
                ]
            )
        ],
        tolerance=tol,
        num_values=2,
    )
    kkt_validator.optimize_queue_order(queue_order=[6, 2, 0, 3, 1, 4, 5])
    kkt_validator = AdaptiveValidatorWrapper(kkt_validator)

    # Manager for convergenced kkt conditions
    kkt_status_manager = ConvergenceConditionManager(
        validator=kkt_validator,
        num_conditions=kkt_validator.get_num_conditions(),
        tolerance=tol
    )

    # Main loop
    start_time = time.perf_counter()
    for counter_main in range(nit):
        if is_constant_scaling and \
                adjust_params.is_to_scale(counter_main):
            scale_prim_dual(msg=f"Var Norm at iteration {counter_main}")
        
        if is_z_scaling and \
                adjust_params.is_to_scale_matrix(counter_main, run_history.get_current_kkt_errors()):
            kkt_errors = run_history.get_current_kkt_errors()
            rescale_z = prim_gap * sqrt(kkt_errors[1] / kkt_errors[0])
            if rescale_z > 1.25:
                scale_variable_z(scale_factor=rescale_z, msg=f"Rescale z at iteration {counter_main}")

        if is_palm:
            with run_history.timer(tag="Step 0 (Q & Lambda)"):
                solve_q_lambda(congestion, r, dt_phi, dx_phi, mu, E, z_fst, z_mid, z_end, beta_fst, beta_mid, beta_end,
                                memo=(memo_A, memo_B),
                                output=(A, B, lambda_c))

        if is_multi_threads:
            with run_history.timer(tag="Step 1 (Lap & SOC-Proj)"):
                thread_lap = threading.Thread(
                    target=solve_laplacian,
                    args=(A, B, lambda_c, mu, E, boundary_time_with_area),
                    kwargs={"output": phi}
                )
                thread_soc = threading.Thread(
                    target=solve_proj_soc,
                    args=(A, B, beta_fst, beta_mid, beta_end),
                    kwargs={
                        "memo": (memo_z_fst, memo_z_mid, memo_z_end, memo_z_mid2),
                        "const_d": constant_d,
                        "scale_z": scale_factor_z,
                        "output": (z_fst, z_mid, z_end)
                    }
                )

                thread_lap.start()
                thread_soc.start()

                thread_lap.join()
                thread_soc.join()
                # phi -= phi.mean()
        else:
            with run_history.timer(tag="Step 1-1 (Laplacian)"):
                solve_laplacian(A, B, lambda_c, mu, E, boundary_time_with_area, output=phi)
                # phi -= phi.mean()
            with run_history.timer(tag="Step 1-2 (SOC-Projection)"):
                solve_proj_soc(A, B, beta_fst, beta_mid, beta_end,
                                memo=(memo_z_fst, memo_z_mid, memo_z_end, memo_z_mid2),
                                const_d=constant_d,
                                scale_z=scale_factor_z,
                                output=(z_fst, z_mid, z_end))

        with run_history.timer(tag="Step 2 (Q & Lambda)"):
            dt_phi = grad_time(phi)
            dx_phi = grad_space(phi)
            solve_q_lambda(congestion, r, dt_phi, dx_phi, mu, E, z_fst, z_mid, z_end, beta_fst, beta_mid, beta_end,
                           memo=(memo_A, memo_B),
                           output=(A, B, lambda_c))

        with run_history.timer(tag="Step 3 (Multiplier)"):
            decouple_spacial(B, scale_z=scale_factor_z, output=memo_z_mid)
            ne.evaluate("mu + tau * (dt_phi - A - lambda_c)", out=mu)
            ne.evaluate("E + tau * (dx_phi - B)", out=E)
            ne.evaluate("beta_fst + tau * (z_fst + scale_factor_z * A - constant_d)", out=beta_fst)
            ne.evaluate("beta_mid + tau * (z_mid - memo_z_mid)", out=beta_mid)
            ne.evaluate("beta_end + tau * (z_end - scale_factor_z * A - constant_d)", out=beta_end)

        # Check KKT
        is_time_used_up = (time.perf_counter() - start_time > time_limit)
        whether_adjust_sigma = adjust_params.is_to_adjust(counter_main) or is_time_used_up

        if whether_adjust_sigma:
            required_conditions = kkt_prim_pos + kkt_dual_pos
        else:
            required_conditions = None
        
        if not check_kkt_step_by_step:

            if whether_adjust_sigma:
                kkt_validator.reset_counter() # Reset the counter to ensure the next validation is performed.
            
            kkt_validation_passed, detailed_info = kkt_validator.validate(required_conditions=required_conditions)
            kkt_errors_list = kkt_collector.get_errors()
            org_kkt_errors, kkt_errors = kkt_errors_list[:, 0], kkt_errors_list[:, 1]

            if whether_adjust_sigma:
                kkt_validator.reset_counter() # Reset the counter to ensure the next validation is performed.

            # Record running history
            run_history.record(
                current_it=counter_main,
                kkt_errors=org_kkt_errors,
            )

            error = max_of_list_with_none(org_kkt_errors[kkt_stop_condition])

            if error is not None:
                kkt_validator.set_error_and_tolerance(error, tol)

            # Show progress (Do not show progress if adjusting sigma for better visualization)
            if error is not None and not whether_adjust_sigma:
                # Compute active and converged conditions
                active_conditions, converged_conditions = kkt_status_manager.compute_conditions(
                    detailed_info, org_kkt_errors, required_conditions
                )

                run_history.show_tol_progress(
                    counter_main,
                    error,
                    active_idx=active_conditions,
                    converged_idx=converged_conditions,
                )
        else:
            kkt_validation_passed, detailed_info = kkt_validator.validate(required_conditions=list(range(kkt_validator.get_num_conditions())))
            kkt_errors_list = kkt_collector.get_errors()
            org_kkt_errors, kkt_errors = kkt_errors_list[:, 0], kkt_errors_list[:, 1]
            trans_cost, lagrangian = objective_functional(
                prim_scale * phi, prim_scale * lambda_c, (dual_scale * r) * boundary_time_with_area, congestion * prim_scale / dual_scale
            )
            
            run_history.record(
                current_it=counter_main,
                kkt_errors=org_kkt_errors,
                history={
                    "Transportation cost": trans_cost,
                    "Objective value": lagrangian,
                }
            )

            error = max_of_list_with_none(org_kkt_errors[kkt_stop_condition])
            run_history.show_tol_progress(counter_main, error)
        
        # Checkpoints
        if tol_checkpoints and error is not None and error <= tol_checkpoints[0]:
            checkpoint_solutions.append(
                recorver_scaled_solution(
                    checkpoint=True,
                    kwargs={
                        "iteration": counter_main,
                        "time": run_history.get_running_time(),
                        "kkt": org_kkt_errors,
                    }
                )
            )
            tol_checkpoints.pop(0)
        
        # Break early
        if kkt_validation_passed or is_time_used_up:
            break

        # Adjust sigma according to original KKT once the scaled KKT is small enough
        max_kkt_errors = max_of_list_with_none(kkt_errors)
        if max_kkt_errors is not None and max_kkt_errors < 5 * tol:
            is_org_kkt = True

        # Update the parameter r
        if whether_adjust_sigma:
            if is_org_kkt:
                prim_error = max_of_list_with_none(org_kkt_errors[kkt_prim_pos]) 
                dual_error = max_of_list_with_none(org_kkt_errors[kkt_dual_pos])
            else:
                prim_error = max_of_list_with_none(kkt_errors[kkt_prim_pos]) 
                dual_error = max_of_list_with_none(kkt_errors[kkt_dual_pos])
            
            prim_dual_gap = prim_error / dual_error
            r_factor = adjust_params.get_updated_value(r, prim_dual_gap) / r
            adjust_penalty(r_factor)

    # Logging running history at the end
    kkt_validation_passed, detailed_info = kkt_validator.validate(required_conditions=list(range(kkt_validator.get_num_conditions())))
    kkt_errors_list = kkt_collector.get_errors()
    org_kkt_errors = kkt_errors_list[:, 0]
    trans_cost, lagrangian = objective_functional(
        prim_scale * phi, prim_scale * lambda_c, (dual_scale * r) * boundary_time_with_area, congestion * prim_scale / dual_scale
    )

    # Record running history
    run_history.record(
        current_it=counter_main,
        kkt_errors=org_kkt_errors,
        history={
            "Transportation cost": trans_cost,
            "Objective value": lagrangian,
        }
    )

    # Ending
    run_history.end()
    recorver_scaled_solution()

    # Print info
    logging.log(LOG_LEVELS["info"],
        "---- Overview of solution ".ljust(42, '-') + "\n"
        f"Congestion norm: {np.linalg.norm(lambda_c - congestion * mu):.2f}\n"
        f"Number of iterations: {counter_main}\n"
        f"Iteration time: {run_history.running_time:.2f}"
    )

    solution = SolutionSocpData(
        phi=phi,
        A=A, 
        B=B,
        lambda_c=lambda_c,
        mu=mu,
        E=E,
        z_fst=z_fst,
        z_mid=z_mid, 
        z_end=z_end,
        beta_fst=beta_fst,
        beta_mid=beta_mid,
        beta_end=beta_end,
        checkpoints=checkpoint_solutions if checkpoint_solutions else None,
    )

    return solution, run_history


# Scalar products
def norm_square_weight(weight, num_avg, a):
    """Scalar product
    """
    return np.sum(ne.evaluate("a ** 2 * weight")) / num_avg

# Differential, averaging and projection operators
def vanilla_grad_time(stepsize_time, mu):
    """Gradient wrt Time, ie temporal derivative
    """
    return np.diff(mu, axis=0) / stepsize_time

def vanilla_div_time(stepsize_time, mu):
    """Adjoint of gradient wrt time
    """
    muSize = mu.shape
    output = np.zeros((muSize[0] + 1, muSize[1]))

    output[1:-1, :] = np.diff(mu, axis=0) / stepsize_time
    output[0, :] = mu[0, :] / stepsize_time
    output[-1, :] = - mu[-1, :] / stepsize_time

    return output

def vanilla_grad_space(n_time, n_triangles, gradient_matrix, mu):
    """Gradient wrt space
    """

    return (
        gradient_matrix
        .dot(mu.transpose())
        .transpose()
        .reshape((n_time + 1, n_triangles, 3))
    )

def vanilla_div_space(n_time, n_vertices, n_triangles, div_matrix, mu):
    """Adjoint of gradient wrt space
    """

    muAux = mu.reshape((n_time + 1, 3 * n_triangles))

    # Transpose and reshape to apply divergenceDMatrix independently for fixed first two indices
    return (
        div_matrix
        .dot(muAux.transpose())
        .transpose()
        .reshape((n_time + 1, n_vertices))
    )

def decouple_spacial(b, scale_z: float = 1.0, output=None):
    """ Copy b to construct decoupled variable which located at Space-Staggered grid
    Convert b in (nTime + 1, nTriangles, 3) to z_mid in (nTime, 2, 3, nTriangles, 3)
    """
    n_time, n_triangles = b.shape[0] - 1, b.shape[1]

    if output is None:
        output = np.zeros((n_time, 2, 3, n_triangles, 3))

    b_aux = (scale_z / sqrt(3.0)) * b

    output[:, 0, :, :, :] = np.broadcast_to(
        np.expand_dims(b_aux[:-1, :, :], axis=1),
        (n_time, 3, n_triangles, 3))

    output[:, 1, :, :, :] = np.broadcast_to(
        np.expand_dims(b_aux[1:, :, :], axis=1),
        (n_time, 3, n_triangles, 3))

    return output

def decouple_adjoin_spacial(x, scale_z: float = 1.0, output=None):
    """ Adjoin of decouple_spacial
    Convert x in (nTime, 2, 3, nTriangles, 3) to b in (nTime + 1, nTriangles, 3)
    """
    n_time, n_triangles = x.shape[0], x.shape[3]

    if output is None:
        output = np.zeros((n_time + 1, n_triangles, 3))

    x_aux = (scale_z / sqrt(3.0)) * np.sum(x, axis=2)

    output[:-1, :, :] = x_aux[:, 0, :, :]
    output[-1, :, :]  = 0.0
    output[1:, :, :] += x_aux[:, 1, :, :]

    return output

def decouple_adjoint_time(x, output=None):
    """Adjoint of L_T
    """

    kernel = np.array([0.5, 0.5])

    # Pad zero at the last line along axis
    pad_width = [(0, 0)] * x.ndim
    pad_width[0] = (0, 1)

    if output is None:
        return correlate1d(np.pad(x, pad_width, mode='constant'), kernel, axis=0, mode='constant', cval=0.0)
    else:
        correlate1d(np.pad(x, pad_width, mode='constant'), kernel, axis=0, mode='constant', cval=0.0, output=output)

def vanilla_solve_laplacian(laplacian_invert, div_time, div_space, weight_time, weight_space, weight_center, eps, A, B, lambda_c, mu, E, boundary_item, output=None):
    if output is None:
        return laplacian_invert(
            div_time(ne.evaluate("(A + lambda_c - mu) * weight_time"))
            + div_space(ne.evaluate("(B - E) * weight_space"))
            - boundary_item)
    else:
        output[:] = laplacian_invert(
            div_time(ne.evaluate("(A + lambda_c - mu) * weight_time"))
            + div_space(ne.evaluate("(B - E) * weight_space"))
            - boundary_item - eps * weight_center * output)

def vanilla_solve_proj_soc(diagonal_to_soc_global, diagonal_to_soc, origin_triangles_global_one, vertex_triangles_global, A, B, beta_fst, beta_mid, beta_end, memo, const_d: float = 1.0, scale_z: float = 1.0, output = None):
    """Projection of SOC

    memo is like (z_fst, z_mid, z_end, z_mid)
    """

    # to project into SOC
    decouple_spacial(B, scale_z=scale_z, output=memo[3])
    memo_z_mid = memo[3]
    ne.evaluate("const_d - scale_z * A - beta_fst", out=memo[0])
    ne.evaluate("diagonal_to_soc_global * (memo_z_mid - beta_mid)", out=memo[1])
    ne.evaluate("const_d + scale_z * A - beta_end", out=memo[2])
    (to_proj_z_fst, to_proj_z_mid, to_proj_z_end) = memo[0:3]

    # Lambda
    ne.evaluate("to_proj_z_mid ** 2", out=memo[3])
    norm_zt = origin_triangles_global_one.dot(
        ne.evaluate(
            "a0 + a1 + a2 + a3 + a4 + a5",
            local_dict={
                "a0": memo[3][:, 0, :, :, 0],
                "a1": memo[3][:, 0, :, :, 1],
                "a2": memo[3][:, 0, :, :, 2],
                "a3": memo[3][:, 1, :, :, 0],
                "a4": memo[3][:, 1, :, :, 1],
                "a5": memo[3][:, 1, :, :, 2],
            }
        ).reshape(-1)
    ).reshape(A.shape)
    ne.evaluate("sqrt(norm_zt + to_proj_z_end ** 2)", out=norm_zt)
    lam = np.clip(ne.evaluate("0.5 * (1.0 + to_proj_z_fst / norm_zt)"), 0.0, 1.0)
    ind = ne.evaluate("lam >= 1.0")
    lam_triangles = np.broadcast_to(
        np.expand_dims(
            ne.evaluate(
                "lam2 / diagonal_to_soc",
                local_dict={
                    "lam2": vertex_triangles_global.dot(lam.reshape(-1)).reshape(diagonal_to_soc.shape),
                    "diagonal_to_soc": diagonal_to_soc
                }),
            axis=[1, -1]),
        beta_mid.shape,
    )

    # Projection into (z_fst, z_mid, z_end)
    if output is None:
        z_fst = np.where(ind, to_proj_z_fst, ne.evaluate("lam * norm_zt"))
        z_end = ne.evaluate("lam * to_proj_z_end")
        z_mid = ne.evaluate("lam_triangles * to_proj_z_mid")

        return z_fst, z_mid, z_end
    else:
        output[0][:] = np.where(ind, to_proj_z_fst, ne.evaluate("lam * norm_zt"))
        ne.evaluate("lam_triangles * to_proj_z_mid", out=output[1])
        ne.evaluate("lam * to_proj_z_end", out=output[2])

def vanilla_solve_q_lambda(scale_z, diagonal_b, congestion, r, dt_phi, dx_phi, mu, E, z_fst, z_mid, z_end, beta_fst, beta_mid, beta_end, memo, output = None):
    """ Step (q, lambdaC) with lambdaC = lambdaC(q)

    memo is like (A, B)
    """
    a_const1 = scale_z * (1.0 + congestion * r)
    a_const2 = 1.0 + 2.0 * scale_z * a_const1
    ne.evaluate("dt_phi + mu", out=memo[0])
    decouple_adjoin_spacial(ne.evaluate("z_mid + beta_mid"), scale_z=scale_z, output=memo[1])
    memo_a, memo_b = memo

    if output is None:
        A = ne.evaluate("(1.0 / a_const2) * memo_a + (a_const1 / a_const2) * (z_end + beta_end - z_fst - beta_fst)")
        B = ne.evaluate("(dx_phi + E + memo_b) / diagonal_b")
        lambdaC = ne.evaluate("(congestion * r / (1. + congestion * r)) * (memo_a - A)")

        return A, B, lambdaC
    else:
        ne.evaluate("(1.0 / a_const2) * memo_a + (a_const1 / a_const2) * (z_end + beta_end - z_fst - beta_fst)", out=output[0])
        A = output[0]
        ne.evaluate("(dx_phi + E + memo_b) / diagonal_b", out=output[1])
        ne.evaluate("(congestion * r / (1. + congestion * r)) * (memo_a - A)", out=output[2])
