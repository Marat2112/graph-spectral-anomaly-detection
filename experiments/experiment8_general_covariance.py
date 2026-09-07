import os
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx

from scipy.integrate import quad
from scipy.optimize import brentq
from sklearn.metrics import (
    r2_score,
    mean_absolute_error,
    mean_squared_error,
)


# ============================================================
# EXPERIMENT 8
# GENERAL GAUSSIAN COVARIANCE
#
# x = mu + eps
# eps ~ N(0, Sigma)
#
# Q = x^T L x
#
# General spectral representation:
#
# Sigma = C C^T
# B = C^T L C
# B = V Lambda V^T
#
# Q = c + sum_i lambda_i chi^2_1(delta_i)
#
# ============================================================


OUTPUT_DIR = "results/experiment_08"

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# ============================================================
# EXPERIMENT SETTINGS
# ============================================================

GRAPH_TYPES = [
    "star",
    "random_geometric",
    "barabasi_albert",
]

WEIGHT_MODES = [
    "unweighted",
    "weighted",
]

NUM_SENSOR_VALUES = [
    15,
    30,
    60,
]

NOISE_MODELS = [
    "iid",
    "heteroscedastic",
    "correlated",
]

NOISE_LEVELS = [
    0.15,
    0.30,
]

ANOMALY_AMPLITUDES = [
    1.0,
    1.75,
    3.0,
]

REPETITIONS = 3

ALPHA = 0.01

CALIBRATION_SAMPLES = 5000
NORMAL_TEST_SAMPLES = 5000
ANOMALY_TEST_SAMPLES = 5000

MASTER_SEED = 20260920


# ============================================================
# NUMERICAL SETTINGS
# ============================================================

EIGENVALUE_TOL = 1e-10

IMHOF_EPSABS = 1e-7
IMHOF_EPSREL = 1e-7
IMHOF_LIMIT = 1000

ROOT_XTOL = 1e-8
ROOT_RTOL = 1e-8

CORRELATION_STRENGTH = 0.45


MASTER_RNG = np.random.default_rng(
    MASTER_SEED
)


# ============================================================
# GRAPH GENERATION
# ============================================================

def generate_graph(
    graph_type,
    n,
    rng
):

    if graph_type == "star":

        G = nx.star_graph(
            n - 1
        )

    elif graph_type == "random_geometric":

        radius = 1.5 * np.sqrt(
            np.log(n) / n
        )

        for _ in range(200):

            seed = int(
                rng.integers(
                    0,
                    2**31 - 1
                )
            )

            G = nx.random_geometric_graph(
                n=n,
                radius=radius,
                seed=seed
            )

            if nx.is_connected(G):
                break

        else:

            raise RuntimeError(
                "Could not generate connected "
                "random geometric graph"
            )

    elif graph_type == "barabasi_albert":

        m = min(
            3,
            n - 1
        )

        seed = int(
            rng.integers(
                0,
                2**31 - 1
            )
        )

        G = nx.barabasi_albert_graph(
            n=n,
            m=m,
            seed=seed
        )

    else:

        raise ValueError(
            f"Unknown graph type: {graph_type}"
        )

    return G


# ============================================================
# EDGE WEIGHTS
# ============================================================

def apply_edge_weights(
    G,
    weight_mode,
    rng
):

    G = G.copy()

    if weight_mode == "unweighted":

        for u, v in G.edges():

            G[u][v]["weight"] = 1.0

    elif weight_mode == "weighted":

        for u, v in G.edges():

            weight = rng.uniform(
                0.5,
                1.5
            )

            G[u][v]["weight"] = float(
                weight
            )

    else:

        raise ValueError(
            f"Unknown weight mode: {weight_mode}"
        )

    return G


# ============================================================
# WEIGHTED LAPLACIAN
# ============================================================

def graph_laplacian(
    G
):

    A = nx.to_numpy_array(
        G,
        weight="weight",
        dtype=float
    )

    weighted_degrees = np.sum(
        A,
        axis=1
    )

    D = np.diag(
        weighted_degrees
    )

    L = D - A

    return (
        A,
        L,
        weighted_degrees
    )


# ============================================================
# GRAPH ENERGY
# ============================================================

def graph_energy_samples(
    X,
    L
):

    return np.einsum(
        "bi,ij,bj->b",
        X,
        L,
        X
    )


# ============================================================
# SYMMETRIC MATRIX SQUARE ROOT
# ============================================================

def symmetric_matrix_sqrt(
    Sigma
):

    eigenvalues, eigenvectors = np.linalg.eigh(
        Sigma
    )

    min_eigenvalue = float(
        np.min(
            eigenvalues
        )
    )

    if min_eigenvalue <= 0:

        raise ValueError(
            "Sigma is not positive definite. "
            f"Minimum eigenvalue = {min_eigenvalue}"
        )

    sqrt_eigenvalues = np.sqrt(
        eigenvalues
    )

    C = (
        eigenvectors
        @
        np.diag(
            sqrt_eigenvalues
        )
        @
        eigenvectors.T
    )

    return C


# ============================================================
# NORMALIZED ADJACENCY
# ============================================================

def normalized_adjacency(
    A
):

    degrees = np.sum(
        A,
        axis=1
    )

    inv_sqrt_degree = np.zeros_like(
        degrees,
        dtype=float
    )

    positive = degrees > 0

    inv_sqrt_degree[
        positive
    ] = (
        1.0
        /
        np.sqrt(
            degrees[
                positive
            ]
        )
    )

    D_inv_sqrt = np.diag(
        inv_sqrt_degree
    )

    S = (
        D_inv_sqrt
        @
        A
        @
        D_inv_sqrt
    )

    return S


# ============================================================
# COVARIANCE MODELS
# ============================================================

def build_covariance(
    noise_model,
    sigma,
    A_graph,
    rng
):

    n = A_graph.shape[0]

    # --------------------------------------------------------
    # IID
    # --------------------------------------------------------

    if noise_model == "iid":

        Sigma = (
            sigma ** 2
            *
            np.eye(n)
        )

    # --------------------------------------------------------
    # HETEROSCEDASTIC
    #
    # Sensor standard deviations vary approximately
    # between 0.6*sigma and 1.4*sigma.
    #
    # Then normalize mean marginal variance back to sigma^2.
    # --------------------------------------------------------

    elif noise_model == "heteroscedastic":

        scales = np.linspace(
            0.6,
            1.4,
            n
        )

        rng.shuffle(
            scales
        )

        variances = (
            sigma ** 2
            *
            scales ** 2
        )

        Sigma = np.diag(
            variances
        )

        mean_variance = np.trace(
            Sigma
        ) / n

        Sigma *= (
            sigma ** 2
            /
            mean_variance
        )

    # --------------------------------------------------------
    # GRAPH-CORRELATED GAUSSIAN NOISE
    #
    # S = normalized adjacency
    #
    # M = (I - rho*S)^(-1)
    #
    # raw covariance = M M^T
    #
    # This guarantees positive definiteness as long as
    # rho is below the inverse spectral radius.
    #
    # Since ||S|| <= 1 and rho = 0.45, this is safe.
    #
    # Finally normalize average marginal variance
    # to sigma^2.
    # --------------------------------------------------------

    elif noise_model == "correlated":

        S = normalized_adjacency(
            A_graph
        )

        I = np.eye(
            n
        )

        M = np.linalg.inv(
            I
            -
            CORRELATION_STRENGTH
            *
            S
        )

        Sigma_raw = (
            M
            @
            M.T
        )

        mean_variance = (
            np.trace(
                Sigma_raw
            )
            /
            n
        )

        Sigma = (
            sigma ** 2
            *
            Sigma_raw
            /
            mean_variance
        )

    else:

        raise ValueError(
            f"Unknown noise model: {noise_model}"
        )

    # Force numerical symmetry

    Sigma = (
        Sigma
        +
        Sigma.T
    ) / 2.0

    return Sigma


# ============================================================
# COVARIANCE DIAGNOSTICS
# ============================================================

def covariance_statistics(
    Sigma
):

    n = Sigma.shape[0]

    variances = np.diag(
        Sigma
    )

    std = np.sqrt(
        variances
    )

    denom = np.outer(
        std,
        std
    )

    Corr = (
        Sigma
        /
        denom
    )

    off_diagonal_mask = ~np.eye(
        n,
        dtype=bool
    )

    mean_abs_correlation = float(
        np.mean(
            np.abs(
                Corr[
                    off_diagonal_mask
                ]
            )
        )
    )

    eigenvalues = np.linalg.eigvalsh(
        Sigma
    )

    condition_number = float(
        np.max(
            eigenvalues
        )
        /
        np.min(
            eigenvalues
        )
    )

    variance_cv = float(
        np.std(
            variances
        )
        /
        np.mean(
            variances
        )
    )

    return {
        "mean_variance":
            float(
                np.mean(
                    variances
                )
            ),

        "variance_cv":
            variance_cv,

        "mean_abs_correlation":
            mean_abs_correlation,

        "sigma_min_eigenvalue":
            float(
                np.min(
                    eigenvalues
                )
            ),

        "sigma_max_eigenvalue":
            float(
                np.max(
                    eigenvalues
                )
            ),

        "sigma_condition_number":
            condition_number,
    }


# ============================================================
# GENERAL SPECTRAL REPRESENTATION
# ============================================================

def general_spectral_decomposition(
    L,
    Sigma
):

    # Sigma = C C^T

    C = symmetric_matrix_sqrt(
        Sigma
    )

    # General quadratic form matrix in standard-normal
    # coordinates.

    B = (
        C.T
        @
        L
        @
        C
    )

    B = (
        B
        +
        B.T
    ) / 2.0

    eigenvalues, eigenvectors = np.linalg.eigh(
        B
    )

    mask = (
        eigenvalues
        >
        EIGENVALUE_TOL
    )

    positive_eigenvalues = (
        eigenvalues[
            mask
        ]
    )

    positive_eigenvectors = (
        eigenvectors[
            :,
            mask
        ]
    )

    return (
        C,
        B,
        positive_eigenvalues,
        positive_eigenvectors
    )


# ============================================================
# H1 NONCENTRAL PARAMETERS
# ============================================================

def h1_parameters(
    mu,
    L,
    C,
    eigenvalues,
    eigenvectors
):

    # g = C^T L mu

    g = (
        C.T
        @
        L
        @
        mu
    )

    # Transform linear term to eigenbasis of B

    h = (
        eigenvectors.T
        @
        g
    )

    noncentralities = (
        h
        /
        eigenvalues
    ) ** 2

    mu_energy = float(
        mu.T
        @
        L
        @
        mu
    )

    completed_square_energy = float(
        np.sum(
            h ** 2
            /
            eigenvalues
        )
    )

    constant_shift = (
        mu_energy
        -
        completed_square_energy
    )

    # Tiny roundoff values should be treated as zero.

    if abs(
        constant_shift
    ) < 1e-10:

        constant_shift = 0.0

    return (
        noncentralities,
        float(
            constant_shift
        )
    )


# ============================================================
# IMHOF SURVIVAL FUNCTION
#
# Distribution:
#
# Q = shift + sum lambda_i chi^2_1(delta_i)
#
# ============================================================

def imhof_sf(
    q,
    weights,
    noncentralities,
    shift=0.0
):

    weights = np.asarray(
        weights,
        dtype=float
    )

    noncentralities = np.asarray(
        noncentralities,
        dtype=float
    )

    effective_q = (
        q
        -
        shift
    )

    if effective_q <= 0:

        return 1.0

    mask = (
        weights
        >
        EIGENVALUE_TOL
    )

    weights = weights[
        mask
    ]

    noncentralities = noncentralities[
        mask
    ]

    if len(weights) == 0:

        return float(
            effective_q < 0
        )

    def integrand(
        u
    ):

        if u == 0.0:

            return 0.0

        wu = (
            weights
            *
            u
        )

        wu2 = (
            wu ** 2
        )

        theta = (
            0.5
            *
            np.sum(
                np.arctan(
                    wu
                )
                +
                noncentralities
                *
                wu
                /
                (
                    1.0
                    +
                    wu2
                )
            )
            -
            0.5
            *
            effective_q
            *
            u
        )

        log_rho = (
            0.25
            *
            np.sum(
                np.log1p(
                    wu2
                )
            )
            +
            0.5
            *
            np.sum(
                noncentralities
                *
                wu2
                /
                (
                    1.0
                    +
                    wu2
                )
            )
        )

        if log_rho > 700:

            return 0.0

        rho = np.exp(
            log_rho
        )

        value = (
            np.sin(
                theta
            )
            /
            (
                u
                *
                rho
            )
        )

        return float(
            value
        )

    with warnings.catch_warnings():

        warnings.simplefilter(
            "ignore"
        )

        integral, integration_error = quad(
            integrand,
            0.0,
            np.inf,
            epsabs=IMHOF_EPSABS,
            epsrel=IMHOF_EPSREL,
            limit=IMHOF_LIMIT
        )

    probability = (
        0.5
        +
        integral
        /
        np.pi
    )

    probability = float(
        np.clip(
            probability,
            0.0,
            1.0
        )
    )

    return probability


# ============================================================
# H0 SURVIVAL
# ============================================================

def h0_spectral_sf(
    q,
    eigenvalues
):

    noncentralities = np.zeros_like(
        eigenvalues
    )

    return imhof_sf(
        q=q,
        weights=eigenvalues,
        noncentralities=noncentralities,
        shift=0.0
    )


# ============================================================
# H1 SURVIVAL
# ============================================================

def h1_spectral_sf(
    q,
    mu,
    L,
    C,
    eigenvalues,
    eigenvectors
):

    (
        noncentralities,
        constant_shift
    ) = h1_parameters(
        mu=mu,
        L=L,
        C=C,
        eigenvalues=eigenvalues,
        eigenvectors=eigenvectors
    )

    probability = imhof_sf(
        q=q,
        weights=eigenvalues,
        noncentralities=noncentralities,
        shift=constant_shift
    )

    return (
        probability,
        constant_shift,
        noncentralities
    )


# ============================================================
# THEORETICAL H0 MOMENTS
# ============================================================

def h0_moments(
    L,
    Sigma
):

    mean = float(
        np.trace(
            L
            @
            Sigma
        )
    )

    LSigma = (
        L
        @
        Sigma
    )

    variance = float(
        2.0
        *
        np.trace(
            LSigma
            @
            LSigma
        )
    )

    return (
        mean,
        variance
    )


# ============================================================
# SPECTRAL THRESHOLD
# ============================================================

def spectral_threshold(
    eigenvalues,
    mean_q,
    variance_q,
    alpha
):

    std_q = np.sqrt(
        max(
            variance_q,
            1e-15
        )
    )

    lower = 0.0

    upper = (
        mean_q
        +
        10.0
        *
        std_q
    )

    if upper <= 0:

        upper = 1.0

    for _ in range(30):

        sf_upper = h0_spectral_sf(
            upper,
            eigenvalues
        )

        if sf_upper < alpha:
            break

        upper *= 2.0

    else:

        raise RuntimeError(
            "Could not bracket spectral threshold."
        )

    def objective(
        q
    ):

        return (
            h0_spectral_sf(
                q,
                eigenvalues
            )
            -
            alpha
        )

    tau = brentq(
        objective,
        lower,
        upper,
        xtol=ROOT_XTOL,
        rtol=ROOT_RTOL,
        maxiter=200
    )

    return float(
        tau
    )


# ============================================================
# GENERATE GAUSSIAN SAMPLES
# ============================================================

def gaussian_samples(
    rng,
    covariance,
    sample_count
):

    n = covariance.shape[0]

    return rng.multivariate_normal(
        mean=np.zeros(n),
        cov=covariance,
        size=sample_count,
        check_valid="raise"
    )


# ============================================================
# REGRESSION METRICS
# ============================================================

def regression_metrics(
    observed,
    predicted
):

    observed = np.asarray(
        observed,
        dtype=float
    )

    predicted = np.asarray(
        predicted,
        dtype=float
    )

    r2 = r2_score(
        observed,
        predicted
    )

    mae = mean_absolute_error(
        observed,
        predicted
    )

    rmse = np.sqrt(
        mean_squared_error(
            observed,
            predicted
        )
    )

    bias = float(
        np.mean(
            observed
            -
            predicted
        )
    )

    max_abs_error = float(
        np.max(
            np.abs(
                observed
                -
                predicted
            )
        )
    )

    return {
        "R2":
            float(
                r2
            ),

        "MAE":
            float(
                mae
            ),

        "RMSE":
            float(
                rmse
            ),

        "bias":
            bias,

        "max_abs_error":
            max_abs_error,
    }


# ============================================================
# PREPARE ONE BASE CONFIGURATION
#
# Graph, weights, Sigma, threshold do not depend on A.
# ============================================================

def prepare_base_configuration(
    graph_type,
    weight_mode,
    n,
    noise_model,
    sigma,
    repetition,
    rng
):

    # --------------------------------------------------------
    # Graph
    # --------------------------------------------------------

    G = generate_graph(
        graph_type,
        n,
        rng
    )

    G = apply_edge_weights(
        G,
        weight_mode,
        rng
    )

    A_graph, L, degrees = graph_laplacian(
        G
    )

    # --------------------------------------------------------
    # Covariance
    # --------------------------------------------------------

    Sigma = build_covariance(
        noise_model,
        sigma,
        A_graph,
        rng
    )

    covariance_stats = covariance_statistics(
        Sigma
    )

    # --------------------------------------------------------
    # General spectral representation
    # --------------------------------------------------------

    (
        C,
        B,
        eigenvalues,
        eigenvectors
    ) = general_spectral_decomposition(
        L,
        Sigma
    )

    # --------------------------------------------------------
    # Exact H0 moments
    # --------------------------------------------------------

    theoretical_mean_q, theoretical_var_q = (
        h0_moments(
            L,
            Sigma
        )
    )

    # Equivalent spectral moments for verification

    spectral_mean_q = float(
        np.sum(
            eigenvalues
        )
    )

    spectral_var_q = float(
        2.0
        *
        np.sum(
            eigenvalues ** 2
        )
    )

    # --------------------------------------------------------
    # Calibration sample
    # --------------------------------------------------------

    calibration_noise = gaussian_samples(
        rng,
        Sigma,
        CALIBRATION_SAMPLES
    )

    q_calibration = graph_energy_samples(
        calibration_noise,
        L
    )

    tau_empirical = float(
        np.quantile(
            q_calibration,
            1.0 - ALPHA
        )
    )

    # --------------------------------------------------------
    # Independent H0 test
    # --------------------------------------------------------

    normal_noise = gaussian_samples(
        rng,
        Sigma,
        NORMAL_TEST_SAMPLES
    )

    q_normal = graph_energy_samples(
        normal_noise,
        L
    )

    pfa_mc_emp_tau = float(
        np.mean(
            q_normal
            >
            tau_empirical
        )
    )

    # --------------------------------------------------------
    # Fully spectral threshold
    # --------------------------------------------------------

    tau_spectral = spectral_threshold(
        eigenvalues=eigenvalues,
        mean_q=theoretical_mean_q,
        variance_q=theoretical_var_q,
        alpha=ALPHA
    )

    pfa_mc_spec_tau = float(
        np.mean(
            q_normal
            >
            tau_spectral
        )
    )

    pfa_theory_emp_tau = h0_spectral_sf(
        tau_empirical,
        eigenvalues
    )

    pfa_theory_spec_tau = h0_spectral_sf(
        tau_spectral,
        eigenvalues
    )

    # --------------------------------------------------------
    # Graph statistics
    # --------------------------------------------------------

    trace_L = float(
        np.trace(
            L
        )
    )

    trace_L2 = float(
        np.trace(
            L
            @
            L
        )
    )

    mean_degree = float(
        np.mean(
            degrees
        )
    )

    degree_cv = float(
        np.std(
            degrees
        )
        /
        mean_degree
    )

    # Original Laplacian eigenvalues only for diagnostics

    L_eigenvalues = np.linalg.eigvalsh(
        L
    )

    positive_L = L_eigenvalues[
        L_eigenvalues
        >
        EIGENVALUE_TOL
    ]

    lambda2_L = (
        float(
            positive_L[0]
        )
        if len(
            positive_L
        ) > 0
        else 0.0
    )

    lambda_max_L = float(
        np.max(
            L_eigenvalues
        )
    )

    return {
        "G":
            G,

        "L":
            L,

        "degrees":
            degrees,

        "Sigma":
            Sigma,

        "C":
            C,

        "B":
            B,

        "eigenvalues":
            eigenvalues,

        "eigenvectors":
            eigenvectors,

        "tau_empirical":
            tau_empirical,

        "tau_spectral":
            tau_spectral,

        "pfa_mc_emp_tau":
            pfa_mc_emp_tau,

        "pfa_mc_spec_tau":
            pfa_mc_spec_tau,

        "pfa_theory_emp_tau":
            pfa_theory_emp_tau,

        "pfa_theory_spec_tau":
            pfa_theory_spec_tau,

        "theoretical_mean_q":
            theoretical_mean_q,

        "theoretical_var_q":
            theoretical_var_q,

        "spectral_mean_q":
            spectral_mean_q,

        "spectral_var_q":
            spectral_var_q,

        "trace_L":
            trace_L,

        "trace_L2":
            trace_L2,

        "mean_degree":
            mean_degree,

        "degree_cv":
            degree_cv,

        "lambda2_L":
            lambda2_L,

        "lambda_max_L":
            lambda_max_L,

        **covariance_stats,
    }


# ============================================================
# RUN ONE ANOMALY AMPLITUDE
# ============================================================

def run_anomaly_condition(
    base,
    graph_type,
    weight_mode,
    n,
    noise_model,
    sigma,
    amplitude,
    repetition,
    rng
):

    L = base[
        "L"
    ]

    Sigma = base[
        "Sigma"
    ]

    C = base[
        "C"
    ]

    eigenvalues = base[
        "eigenvalues"
    ]

    eigenvectors = base[
        "eigenvectors"
    ]

    tau_empirical = base[
        "tau_empirical"
    ]

    tau_spectral = base[
        "tau_spectral"
    ]

    # --------------------------------------------------------
    # Monte Carlo H1
    # --------------------------------------------------------

    anomaly_nodes = rng.integers(
        low=0,
        high=n,
        size=ANOMALY_TEST_SAMPLES
    )

    noise = gaussian_samples(
        rng,
        Sigma,
        ANOMALY_TEST_SAMPLES
    )

    X = noise.copy()

    X[
        np.arange(
            ANOMALY_TEST_SAMPLES
        ),
        anomaly_nodes
    ] += amplitude

    q_anomaly = graph_energy_samples(
        X,
        L
    )

    pd_mc_emp_tau = float(
        np.mean(
            q_anomaly
            >
            tau_empirical
        )
    )

    pd_mc_spec_tau = float(
        np.mean(
            q_anomaly
            >
            tau_spectral
        )
    )

    # --------------------------------------------------------
    # Full spectral H1
    #
    # Uniformly random anomaly node:
    #
    # P_D = (1/n) sum_k P_D(k)
    # --------------------------------------------------------

    pd_nodes_emp_tau = []
    pd_nodes_spec_tau = []

    constant_shifts = []
    max_noncentralities = []

    for k in range(n):

        mu = np.zeros(
            n,
            dtype=float
        )

        mu[k] = amplitude

        (
            pd_emp_k,
            shift_emp,
            nc_emp
        ) = h1_spectral_sf(
            q=tau_empirical,
            mu=mu,
            L=L,
            C=C,
            eigenvalues=eigenvalues,
            eigenvectors=eigenvectors
        )

        (
            pd_spec_k,
            shift_spec,
            nc_spec
        ) = h1_spectral_sf(
            q=tau_spectral,
            mu=mu,
            L=L,
            C=C,
            eigenvalues=eigenvalues,
            eigenvectors=eigenvectors
        )

        pd_nodes_emp_tau.append(
            pd_emp_k
        )

        pd_nodes_spec_tau.append(
            pd_spec_k
        )

        constant_shifts.append(
            shift_spec
        )

        max_noncentralities.append(
            np.max(
                nc_spec
            )
            if len(
                nc_spec
            ) > 0
            else 0.0
        )

    pd_theory_emp_tau = float(
        np.mean(
            pd_nodes_emp_tau
        )
    )

    pd_theory_spec_tau = float(
        np.mean(
            pd_nodes_spec_tau
        )
    )

    tau_relative_error = (
        tau_spectral
        -
        tau_empirical
    ) / tau_empirical

    result = {
        "graph_type":
            graph_type,

        "weight_mode":
            weight_mode,

        "n":
            n,

        "noise_model":
            noise_model,

        "sigma":
            sigma,

        "amplitude":
            amplitude,

        "repetition":
            repetition,

        "mean_degree":
            base[
                "mean_degree"
            ],

        "degree_cv":
            base[
                "degree_cv"
            ],

        "trace_L":
            base[
                "trace_L"
            ],

        "trace_L2":
            base[
                "trace_L2"
            ],

        "lambda2_L":
            base[
                "lambda2_L"
            ],

        "lambda_max_L":
            base[
                "lambda_max_L"
            ],

        "mean_variance":
            base[
                "mean_variance"
            ],

        "variance_cv":
            base[
                "variance_cv"
            ],

        "mean_abs_correlation":
            base[
                "mean_abs_correlation"
            ],

        "sigma_condition_number":
            base[
                "sigma_condition_number"
            ],

        "theoretical_mean_q":
            base[
                "theoretical_mean_q"
            ],

        "spectral_mean_q":
            base[
                "spectral_mean_q"
            ],

        "mean_q_relative_difference":
            (
                base[
                    "spectral_mean_q"
                ]
                -
                base[
                    "theoretical_mean_q"
                ]
            )
            /
            max(
                abs(
                    base[
                        "theoretical_mean_q"
                    ]
                ),
                1e-15
            ),

        "theoretical_var_q":
            base[
                "theoretical_var_q"
            ],

        "spectral_var_q":
            base[
                "spectral_var_q"
            ],

        "var_q_relative_difference":
            (
                base[
                    "spectral_var_q"
                ]
                -
                base[
                    "theoretical_var_q"
                ]
            )
            /
            max(
                abs(
                    base[
                        "theoretical_var_q"
                    ]
                ),
                1e-15
            ),

        "tau_empirical":
            tau_empirical,

        "tau_spectral":
            tau_spectral,

        "tau_relative_error":
            tau_relative_error,

        "pfa_mc_emp_tau":
            base[
                "pfa_mc_emp_tau"
            ],

        "pfa_mc_spec_tau":
            base[
                "pfa_mc_spec_tau"
            ],

        "pfa_theory_emp_tau":
            base[
                "pfa_theory_emp_tau"
            ],

        "pfa_theory_spec_tau":
            base[
                "pfa_theory_spec_tau"
            ],

        "pd_mc_emp_tau":
            pd_mc_emp_tau,

        "pd_theory_emp_tau":
            pd_theory_emp_tau,

        "pd_error_emp_tau":
            (
                pd_mc_emp_tau
                -
                pd_theory_emp_tau
            ),

        "pd_mc_spec_tau":
            pd_mc_spec_tau,

        "pd_theory_spec_tau":
            pd_theory_spec_tau,

        "pd_error_spec_tau":
            (
                pd_mc_spec_tau
                -
                pd_theory_spec_tau
            ),

        "max_abs_constant_shift":
            float(
                np.max(
                    np.abs(
                        constant_shifts
                    )
                )
            ),

        "mean_abs_constant_shift":
            float(
                np.mean(
                    np.abs(
                        constant_shifts
                    )
                )
            ),

        "mean_max_noncentrality":
            float(
                np.mean(
                    max_noncentralities
                )
            ),
    }

    return result


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print(
        "===================================================="
    )

    print(
        "EXPERIMENT 8"
    )

    print(
        "GENERAL GAUSSIAN COVARIANCE"
    )

    print(
        "===================================================="
    )

    base_total = (
        len(
            GRAPH_TYPES
        )
        *
        len(
            WEIGHT_MODES
        )
        *
        len(
            NUM_SENSOR_VALUES
        )
        *
        len(
            NOISE_MODELS
        )
        *
        len(
            NOISE_LEVELS
        )
        *
        REPETITIONS
    )

    total_conditions = (
        base_total
        *
        len(
            ANOMALY_AMPLITUDES
        )
    )

    print(
        f"Base configurations: {base_total}"
    )

    print(
        f"Anomaly conditions: {total_conditions}"
    )

    print()

    rows = []

    base_counter = 0

    for graph_type in GRAPH_TYPES:

        for weight_mode in WEIGHT_MODES:

            for n in NUM_SENSOR_VALUES:

                for noise_model in NOISE_MODELS:

                    for sigma in NOISE_LEVELS:

                        for repetition in range(
                            REPETITIONS
                        ):

                            base_counter += 1

                            seed = int(
                                MASTER_RNG.integers(
                                    0,
                                    2**31 - 1
                                )
                            )

                            rng = np.random.default_rng(
                                seed
                            )

                            print()
                            print(
                                f"BASE "
                                f"[{base_counter}/{base_total}]"
                            )

                            print(
                                f"graph={graph_type}, "
                                f"weights={weight_mode}, "
                                f"n={n}, "
                                f"noise={noise_model}, "
                                f"sigma={sigma}, "
                                f"rep={repetition}"
                            )

                            base = (
                                prepare_base_configuration(
                                    graph_type=graph_type,
                                    weight_mode=weight_mode,
                                    n=n,
                                    noise_model=noise_model,
                                    sigma=sigma,
                                    repetition=repetition,
                                    rng=rng
                                )
                            )

                            for amplitude in (
                                ANOMALY_AMPLITUDES
                            ):

                                print(
                                    f"    A={amplitude}"
                                )

                                result = (
                                    run_anomaly_condition(
                                        base=base,
                                        graph_type=graph_type,
                                        weight_mode=weight_mode,
                                        n=n,
                                        noise_model=noise_model,
                                        sigma=sigma,
                                        amplitude=amplitude,
                                        repetition=repetition,
                                        rng=rng
                                    )
                                )

                                rows.append(
                                    result
                                )

    # ========================================================
    # RAW RESULTS
    # ========================================================

    raw_df = pd.DataFrame(
        rows
    )

    raw_df.to_csv(
        os.path.join(
            OUTPUT_DIR,
            "experiment8_raw.csv"
        ),
        index=False
    )

    # ========================================================
    # AGGREGATION
    # ========================================================

    group_columns = [
        "graph_type",
        "weight_mode",
        "n",
        "noise_model",
        "sigma",
        "amplitude",
    ]

    summary_df = (
        raw_df
        .groupby(
            group_columns,
            as_index=False
        )
        .agg(
            tau_empirical_mean=(
                "tau_empirical",
                "mean"
            ),

            tau_spectral_mean=(
                "tau_spectral",
                "mean"
            ),

            tau_relative_error_mean=(
                "tau_relative_error",
                "mean"
            ),

            pfa_mc_emp_mean=(
                "pfa_mc_emp_tau",
                "mean"
            ),

            pfa_mc_spec_mean=(
                "pfa_mc_spec_tau",
                "mean"
            ),

            pfa_theory_emp_mean=(
                "pfa_theory_emp_tau",
                "mean"
            ),

            pfa_theory_spec_mean=(
                "pfa_theory_spec_tau",
                "mean"
            ),

            pd_mc_emp_mean=(
                "pd_mc_emp_tau",
                "mean"
            ),

            pd_theory_emp_mean=(
                "pd_theory_emp_tau",
                "mean"
            ),

            pd_mc_spec_mean=(
                "pd_mc_spec_tau",
                "mean"
            ),

            pd_theory_spec_mean=(
                "pd_theory_spec_tau",
                "mean"
            ),

            variance_cv_mean=(
                "variance_cv",
                "mean"
            ),

            mean_abs_correlation_mean=(
                "mean_abs_correlation",
                "mean"
            ),

            sigma_condition_number_mean=(
                "sigma_condition_number",
                "mean"
            ),

            max_abs_constant_shift=(
                "max_abs_constant_shift",
                "max"
            ),
        )
    )

    summary_df.to_csv(
        os.path.join(
            OUTPUT_DIR,
            "experiment8_summary.csv"
        ),
        index=False
    )

    # ========================================================
    # GLOBAL METRICS
    # ========================================================

    empirical_tau_metrics = regression_metrics(
        summary_df[
            "pd_mc_emp_mean"
        ],
        summary_df[
            "pd_theory_emp_mean"
        ]
    )

    fully_spectral_metrics = regression_metrics(
        summary_df[
            "pd_mc_spec_mean"
        ],
        summary_df[
            "pd_theory_spec_mean"
        ]
    )

    metrics_df = pd.DataFrame(
        [
            {
                "method":
                    "general_spectral_empirical_tau",

                **empirical_tau_metrics
            },

            {
                "method":
                    "general_fully_spectral",

                **fully_spectral_metrics
            },
        ]
    )

    metrics_df.to_csv(
        os.path.join(
            OUTPUT_DIR,
            "experiment8_metrics.csv"
        ),
        index=False
    )

    # ========================================================
    # METRICS BY NOISE MODEL
    # ========================================================

    noise_rows = []

    for noise_model in NOISE_MODELS:

        sub = summary_df[
            summary_df[
                "noise_model"
            ]
            ==
            noise_model
        ]

        metrics = regression_metrics(
            sub[
                "pd_mc_spec_mean"
            ],
            sub[
                "pd_theory_spec_mean"
            ]
        )

        noise_rows.append(
            {
                "noise_model":
                    noise_model,

                **metrics
            }
        )

    metrics_noise_df = pd.DataFrame(
        noise_rows
    )

    metrics_noise_df.to_csv(
        os.path.join(
            OUTPUT_DIR,
            "experiment8_metrics_by_noise.csv"
        ),
        index=False
    )

    # ========================================================
    # METRICS BY GRAPH
    # ========================================================

    graph_rows = []

    for graph_type in GRAPH_TYPES:

        sub = summary_df[
            summary_df[
                "graph_type"
            ]
            ==
            graph_type
        ]

        metrics = regression_metrics(
            sub[
                "pd_mc_spec_mean"
            ],
            sub[
                "pd_theory_spec_mean"
            ]
        )

        graph_rows.append(
            {
                "graph_type":
                    graph_type,

                **metrics
            }
        )

    metrics_graph_df = pd.DataFrame(
        graph_rows
    )

    metrics_graph_df.to_csv(
        os.path.join(
            OUTPUT_DIR,
            "experiment8_metrics_by_graph.csv"
        ),
        index=False
    )

    # ========================================================
    # METRICS BY WEIGHT MODE
    # ========================================================

    weight_rows = []

    for weight_mode in WEIGHT_MODES:

        sub = summary_df[
            summary_df[
                "weight_mode"
            ]
            ==
            weight_mode
        ]

        metrics = regression_metrics(
            sub[
                "pd_mc_spec_mean"
            ],
            sub[
                "pd_theory_spec_mean"
            ]
        )

        weight_rows.append(
            {
                "weight_mode":
                    weight_mode,

                **metrics
            }
        )

    metrics_weight_df = pd.DataFrame(
        weight_rows
    )

    metrics_weight_df.to_csv(
        os.path.join(
            OUTPUT_DIR,
            "experiment8_metrics_by_weight.csv"
        ),
        index=False
    )

    # ========================================================
    # H0 / THRESHOLD DIAGNOSTICS
    # ========================================================

    diagnostics = pd.DataFrame(
        [
            {
                "mean_mc_PFA_emp_tau":
                    float(
                        raw_df[
                            "pfa_mc_emp_tau"
                        ].mean()
                    ),

                "mean_theory_PFA_emp_tau":
                    float(
                        raw_df[
                            "pfa_theory_emp_tau"
                        ].mean()
                    ),

                "mean_mc_PFA_spec_tau":
                    float(
                        raw_df[
                            "pfa_mc_spec_tau"
                        ].mean()
                    ),

                "mean_theory_PFA_spec_tau":
                    float(
                        raw_df[
                            "pfa_theory_spec_tau"
                        ].mean()
                    ),

                "mean_tau_relative_error":
                    float(
                        raw_df[
                            "tau_relative_error"
                        ].mean()
                    ),

                "mean_abs_tau_relative_error":
                    float(
                        np.mean(
                            np.abs(
                                raw_df[
                                    "tau_relative_error"
                                ]
                            )
                        )
                    ),

                "max_abs_tau_relative_error":
                    float(
                        np.max(
                            np.abs(
                                raw_df[
                                    "tau_relative_error"
                                ]
                            )
                        )
                    ),

                "max_abs_mean_moment_difference":
                    float(
                        np.max(
                            np.abs(
                                raw_df[
                                    "mean_q_relative_difference"
                                ]
                            )
                        )
                    ),

                "max_abs_variance_moment_difference":
                    float(
                        np.max(
                            np.abs(
                                raw_df[
                                    "var_q_relative_difference"
                                ]
                            )
                        )
                    ),

                "max_abs_constant_shift":
                    float(
                        raw_df[
                            "max_abs_constant_shift"
                        ].max()
                    ),
            }
        ]
    )

    diagnostics.to_csv(
        os.path.join(
            OUTPUT_DIR,
            "experiment8_diagnostics.csv"
        ),
        index=False
    )

    # ========================================================
    # PLOT 1
    # FULL SPECTRAL OBSERVED VS THEORY
    # ========================================================

    plt.figure(
        figsize=(7, 7)
    )

    plt.scatter(
        summary_df[
            "pd_mc_spec_mean"
        ],
        summary_df[
            "pd_theory_spec_mean"
        ],
        alpha=0.65
    )

    plt.plot(
        [0, 1],
        [0, 1],
        linestyle="--"
    )

    plt.xlabel(
        "Monte Carlo P_D"
    )

    plt.ylabel(
        "General spectral P_D"
    )

    plt.title(
        "Experiment 8: general covariance"
    )

    plt.grid(
        alpha=0.25
    )

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            OUTPUT_DIR,
            "observed_vs_general_spectral.png"
        ),
        dpi=300
    )

    plt.close()

    # ========================================================
    # PLOT 2
    # ERROR VS CORRELATION
    # ========================================================

    errors = (
        summary_df[
            "pd_mc_spec_mean"
        ]
        -
        summary_df[
            "pd_theory_spec_mean"
        ]
    )

    plt.figure(
        figsize=(8, 6)
    )

    plt.scatter(
        summary_df[
            "mean_abs_correlation_mean"
        ],
        errors,
        alpha=0.65
    )

    plt.axhline(
        0.0,
        linestyle="--"
    )

    plt.xlabel(
        "Mean absolute noise correlation"
    )

    plt.ylabel(
        "Monte Carlo P_D - spectral P_D"
    )

    plt.title(
        "Prediction error vs noise correlation"
    )

    plt.grid(
        alpha=0.25
    )

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            OUTPUT_DIR,
            "error_vs_noise_correlation.png"
        ),
        dpi=300
    )

    plt.close()

    # ========================================================
    # PRINT FINAL RESULTS
    # ========================================================

    print()
    print(
        "===================================================="
    )

    print(
        "GLOBAL METRICS"
    )

    print(
        "===================================================="
    )

    print(
        metrics_df.to_string(
            index=False
        )
    )

    print()
    print(
        "===================================================="
    )

    print(
        "METRICS BY NOISE MODEL"
    )

    print(
        "===================================================="
    )

    print(
        metrics_noise_df.to_string(
            index=False
        )
    )

    print()
    print(
        "===================================================="
    )

    print(
        "METRICS BY GRAPH"
    )

    print(
        "===================================================="
    )

    print(
        metrics_graph_df.to_string(
            index=False
        )
    )

    print()
    print(
        "===================================================="
    )

    print(
        "METRICS BY WEIGHT MODE"
    )

    print(
        "===================================================="
    )

    print(
        metrics_weight_df.to_string(
            index=False
        )
    )

    print()
    print(
        "===================================================="
    )

    print(
        "DIAGNOSTICS"
    )

    print(
        "===================================================="
    )

    print(
        diagnostics.to_string(
            index=False
        )
    )

    print()
    print(
        "===================================================="
    )

    print(
        "EXPERIMENT 8 COMPLETED"
    )

    print(
        "===================================================="
    )

    print(
        f"Results saved to: {OUTPUT_DIR}/"
    )


if __name__ == "__main__":
    main()
