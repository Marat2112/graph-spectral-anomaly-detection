import os
import warnings
from collections import deque

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
# EXPERIMENT 9
# SIGNAL GEOMETRY AND DETECTABILITY
#
# x = s + eps
# eps ~ N(0, Sigma)
#
# Q = x^T L x
#
# All anomaly signals have the SAME Euclidean norm:
#
# ||s||_2 = A
#
# but different graph energy:
#
# E_G(s) = s^T L s
#
# ============================================================


OUTPUT_DIR = "results/experiment_09"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# SETTINGS
# ============================================================

GRAPH_TYPES = [
    "star",
    "random_geometric",
    "barabasi_albert",
]

NUM_SENSOR_VALUES = [
    20,
    40,
    60,
]

NOISE_MODELS = [
    "iid",
    "correlated",
]

NOISE_LEVELS = [
    0.15,
    0.30,
]

SIGNAL_AMPLITUDES = [
    0.75,
    1.50,
    3.00,
]

SIGNAL_PATTERNS = [
    "constant",
    "fiedler",
    "cluster",
    "point",
    "edge_bipolar",
    "high_frequency",
]

REPETITIONS = 4

ALPHA = 0.01

CALIBRATION_SAMPLES = 5000
NORMAL_TEST_SAMPLES = 5000
ANOMALY_TEST_SAMPLES = 5000

CORRELATION_STRENGTH = 0.45

MASTER_SEED = 20260925

EIGENVALUE_TOL = 1e-10

IMHOF_EPSABS = 1e-7
IMHOF_EPSREL = 1e-7
IMHOF_LIMIT = 1000

MASTER_RNG = np.random.default_rng(
    MASTER_SEED
)


# ============================================================
# GRAPH GENERATION
# ============================================================

def generate_graph(graph_type, n, rng):

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
                "random geometric graph."
            )

    elif graph_type == "barabasi_albert":

        seed = int(
            rng.integers(
                0,
                2**31 - 1
            )
        )

        G = nx.barabasi_albert_graph(
            n=n,
            m=3,
            seed=seed
        )

    else:

        raise ValueError(
            f"Unknown graph type: {graph_type}"
        )

    if not nx.is_connected(G):

        raise RuntimeError(
            "Generated graph is disconnected."
        )

    return G


# ============================================================
# GRAPH MATRICES
# ============================================================

def graph_matrices(G):

    A = nx.to_numpy_array(
        G,
        dtype=float
    )

    degree = np.sum(
        A,
        axis=1
    )

    D = np.diag(
        degree
    )

    L = D - A

    return A, L, degree


# ============================================================
# GRAPH ENERGY
# ============================================================

def graph_energy_samples(X, L):

    return np.einsum(
        "bi,ij,bj->b",
        X,
        L,
        X
    )


def signal_graph_energy(s, L):

    return float(
        s.T
        @ L
        @ s
    )


# ============================================================
# COVARIANCE
# ============================================================

def normalized_adjacency(A):

    d = np.sum(
        A,
        axis=1
    )

    inv_sqrt_d = np.zeros_like(
        d,
        dtype=float
    )

    mask = d > 0

    inv_sqrt_d[mask] = (
        1.0
        /
        np.sqrt(
            d[mask]
        )
    )

    D_inv = np.diag(
        inv_sqrt_d
    )

    return (
        D_inv
        @ A
        @ D_inv
    )


def build_covariance(
    noise_model,
    sigma,
    A_graph
):

    n = A_graph.shape[0]

    if noise_model == "iid":

        Sigma = (
            sigma ** 2
            *
            np.eye(n)
        )

    elif noise_model == "correlated":

        S = normalized_adjacency(
            A_graph
        )

        M = np.linalg.inv(
            np.eye(n)
            -
            CORRELATION_STRENGTH
            *
            S
        )

        Sigma_raw = (
            M
            @ M.T
        )

        mean_variance = (
            np.trace(Sigma_raw)
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

    Sigma = (
        Sigma
        +
        Sigma.T
    ) / 2.0

    return Sigma


# ============================================================
# SYMMETRIC MATRIX SQRT
# ============================================================

def symmetric_sqrt(Sigma):

    values, vectors = np.linalg.eigh(
        Sigma
    )

    if np.min(values) <= 0:

        raise RuntimeError(
            "Covariance matrix is not "
            "positive definite."
        )

    C = (
        vectors
        @ np.diag(
            np.sqrt(values)
        )
        @ vectors.T
    )

    return C


# ============================================================
# GENERAL SPECTRAL REPRESENTATION
# ============================================================

def spectral_model(L, Sigma):

    C = symmetric_sqrt(
        Sigma
    )

    B = (
        C.T
        @ L
        @ C
    )

    B = (
        B
        +
        B.T
    ) / 2.0

    values, vectors = np.linalg.eigh(
        B
    )

    positive = (
        values
        >
        EIGENVALUE_TOL
    )

    lambda_positive = values[
        positive
    ]

    V_positive = vectors[
        :,
        positive
    ]

    return (
        C,
        B,
        lambda_positive,
        V_positive
    )


# ============================================================
# IMHOF SURVIVAL FUNCTION
# ============================================================

def imhof_sf(
    q,
    weights,
    noncentralities
):

    weights = np.asarray(
        weights,
        dtype=float
    )

    noncentralities = np.asarray(
        noncentralities,
        dtype=float
    )

    if q <= 0:
        return 1.0

    def integrand(u):

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
                np.arctan(wu)
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
            q
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

        return float(
            np.sin(theta)
            /
            (
                u
                *
                rho
            )
        )

    with warnings.catch_warnings():

        warnings.simplefilter(
            "ignore"
        )

        value, error = quad(
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
        value
        /
        np.pi
    )

    return float(
        np.clip(
            probability,
            0.0,
            1.0
        )
    )


# ============================================================
# H0 THEORY
# ============================================================

def h0_sf(
    q,
    eigenvalues
):

    return imhof_sf(
        q,
        eigenvalues,
        np.zeros_like(
            eigenvalues
        )
    )


# ============================================================
# H1 THEORY FOR ARBITRARY SIGNAL s
# ============================================================

def h1_noncentralities(
    signal,
    C,
    eigenvectors
):

    # signal = C m

    m = np.linalg.solve(
        C,
        signal
    )

    # coordinates in eigenbasis of
    # B = C^T L C

    r = (
        eigenvectors.T
        @ m
    )

    delta = (
        r ** 2
    )

    return delta


def h1_sf(
    q,
    signal,
    C,
    eigenvalues,
    eigenvectors
):

    delta = h1_noncentralities(
        signal,
        C,
        eigenvectors
    )

    return imhof_sf(
        q,
        eigenvalues,
        delta
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
            @ Sigma
        )
    )

    LS = (
        L
        @ Sigma
    )

    variance = float(
        2.0
        *
        np.trace(
            LS
            @ LS
        )
    )

    return mean, variance


# ============================================================
# SPECTRAL THRESHOLD
# ============================================================

def spectral_threshold(
    eigenvalues,
    mean_q,
    variance_q
):

    sd_q = np.sqrt(
        variance_q
    )

    lower = 0.0

    upper = (
        mean_q
        +
        10.0
        *
        sd_q
    )

    for _ in range(30):

        if (
            h0_sf(
                upper,
                eigenvalues
            )
            <
            ALPHA
        ):
            break

        upper *= 2.0

    else:

        raise RuntimeError(
            "Could not bracket threshold."
        )

    def objective(q):

        return (
            h0_sf(
                q,
                eigenvalues
            )
            -
            ALPHA
        )

    return float(
        brentq(
            objective,
            lower,
            upper,
            xtol=1e-8,
            rtol=1e-8,
            maxiter=200
        )
    )


# ============================================================
# NORMALIZE SIGNAL TO ||s|| = amplitude
# ============================================================

def normalize_signal(
    signal,
    amplitude
):

    norm = np.linalg.norm(
        signal
    )

    if norm <= 1e-15:

        raise ValueError(
            "Cannot normalize zero signal."
        )

    return (
        amplitude
        *
        signal
        /
        norm
    )


# ============================================================
# CONNECTED CLUSTER
# ============================================================

def connected_cluster_nodes(
    G,
    start_node,
    cluster_size
):

    visited = {
        start_node
    }

    queue = deque(
        [start_node]
    )

    order = []

    while queue and (
        len(order)
        <
        cluster_size
    ):

        u = queue.popleft()

        order.append(
            u
        )

        for v in G.neighbors(u):

            if v not in visited:

                visited.add(v)

                queue.append(v)

    return order[
        :cluster_size
    ]


# ============================================================
# CONSTRUCT SIGNAL
# ============================================================

def construct_signal(
    pattern,
    amplitude,
    G,
    L,
    rng
):

    n = L.shape[0]

    eigenvalues_L, eigenvectors_L = (
        np.linalg.eigh(
            L
        )
    )

    # --------------------------------------------------------
    # CONSTANT
    # --------------------------------------------------------

    if pattern == "constant":

        base = np.ones(
            n
        )

        metadata = {
            "signal_location":
                "all"
        }

    # --------------------------------------------------------
    # FIEDLER / LOW FREQUENCY
    # --------------------------------------------------------

    elif pattern == "fiedler":

        base = (
            eigenvectors_L[
                :,
                1
            ]
        )

        metadata = {
            "signal_location":
                "fiedler"
        }

    # --------------------------------------------------------
    # CONNECTED CLUSTER
    # --------------------------------------------------------

    elif pattern == "cluster":

        start = int(
            rng.integers(
                0,
                n
            )
        )

        cluster_size = max(
            2,
            int(
                round(
                    0.20
                    *
                    n
                )
            )
        )

        nodes = connected_cluster_nodes(
            G,
            start,
            cluster_size
        )

        base = np.zeros(
            n
        )

        base[
            nodes
        ] = 1.0

        metadata = {
            "signal_location":
                ";".join(
                    map(
                        str,
                        nodes
                    )
                )
        }

    # --------------------------------------------------------
    # POINT
    # --------------------------------------------------------

    elif pattern == "point":

        k = int(
            rng.integers(
                0,
                n
            )
        )

        base = np.zeros(
            n
        )

        base[k] = 1.0

        metadata = {
            "signal_location":
                str(k)
        }

    # --------------------------------------------------------
    # TWO NEIGHBOURS, OPPOSITE SIGN
    # --------------------------------------------------------

    elif pattern == "edge_bipolar":

        edges = list(
            G.edges()
        )

        edge_index = int(
            rng.integers(
                0,
                len(edges)
            )
        )

        u, v = edges[
            edge_index
        ]

        base = np.zeros(
            n
        )

        base[u] = 1.0
        base[v] = -1.0

        metadata = {
            "signal_location":
                f"{u}-{v}"
        }

    # --------------------------------------------------------
    # HIGHEST GRAPH FREQUENCY
    # --------------------------------------------------------

    elif pattern == "high_frequency":

        base = (
            eigenvectors_L[
                :,
                -1
            ]
        )

        metadata = {
            "signal_location":
                "lambda_max"
        }

    else:

        raise ValueError(
            f"Unknown signal pattern: {pattern}"
        )

    signal = normalize_signal(
        base,
        amplitude
    )

    return (
        signal,
        metadata,
        eigenvalues_L
    )


# ============================================================
# METRICS
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

    return {
        "R2":
            float(
                r2_score(
                    observed,
                    predicted
                )
            ),

        "MAE":
            float(
                mean_absolute_error(
                    observed,
                    predicted
                )
            ),

        "RMSE":
            float(
                np.sqrt(
                    mean_squared_error(
                        observed,
                        predicted
                    )
                )
            ),

        "bias":
            float(
                np.mean(
                    observed
                    -
                    predicted
                )
            ),

        "max_abs_error":
            float(
                np.max(
                    np.abs(
                        observed
                        -
                        predicted
                    )
                )
            ),
    }


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print(
        "================================================"
    )
    print(
        "EXPERIMENT 9"
    )
    print(
        "SIGNAL GEOMETRY AND DETECTABILITY"
    )
    print(
        "================================================"
    )

    rows = []

    base_total = (
        len(GRAPH_TYPES)
        *
        len(NUM_SENSOR_VALUES)
        *
        len(NOISE_MODELS)
        *
        len(NOISE_LEVELS)
        *
        REPETITIONS
    )

    base_counter = 0

    for graph_type in GRAPH_TYPES:

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

                        print(
                            f"\nBASE "
                            f"[{base_counter}/{base_total}] "
                            f"{graph_type}, "
                            f"n={n}, "
                            f"noise={noise_model}, "
                            f"sigma={sigma}, "
                            f"rep={repetition}"
                        )

                        G = generate_graph(
                            graph_type,
                            n,
                            rng
                        )

                        (
                            A_graph,
                            L,
                            degree
                        ) = graph_matrices(
                            G
                        )

                        Sigma = build_covariance(
                            noise_model,
                            sigma,
                            A_graph
                        )

                        (
                            C,
                            B,
                            spectral_eigenvalues,
                            spectral_eigenvectors
                        ) = spectral_model(
                            L,
                            Sigma
                        )

                        (
                            mean_q,
                            variance_q
                        ) = h0_moments(
                            L,
                            Sigma
                        )

                        tau = spectral_threshold(
                            spectral_eigenvalues,
                            mean_q,
                            variance_q
                        )

                        # Independent H0 Monte Carlo

                        normal_noise = (
                            rng.multivariate_normal(
                                mean=np.zeros(n),
                                cov=Sigma,
                                size=NORMAL_TEST_SAMPLES
                            )
                        )

                        q_normal = graph_energy_samples(
                            normal_noise,
                            L
                        )

                        pfa_mc = float(
                            np.mean(
                                q_normal
                                >
                                tau
                            )
                        )

                        pfa_theory = h0_sf(
                            tau,
                            spectral_eigenvalues
                        )

                        for amplitude in (
                            SIGNAL_AMPLITUDES
                        ):

                            for pattern in (
                                SIGNAL_PATTERNS
                            ):

                                print(
                                    f"    A={amplitude}, "
                                    f"pattern={pattern}"
                                )

                                (
                                    signal,
                                    metadata,
                                    eigenvalues_L
                                ) = construct_signal(
                                    pattern,
                                    amplitude,
                                    G,
                                    L,
                                    rng
                                )

                                signal_norm = float(
                                    np.linalg.norm(
                                        signal
                                    )
                                )

                                graph_energy = (
                                    signal_graph_energy(
                                        signal,
                                        L
                                    )
                                )

                                rayleigh = (
                                    graph_energy
                                    /
                                    (
                                        signal_norm ** 2
                                    )
                                )

                                lambda2 = float(
                                    eigenvalues_L[1]
                                )

                                lambda_max = float(
                                    eigenvalues_L[-1]
                                )

                                # ----------------------------
                                # Monte Carlo H1
                                # ----------------------------

                                anomaly_noise = (
                                    rng.multivariate_normal(
                                        mean=np.zeros(n),
                                        cov=Sigma,
                                        size=ANOMALY_TEST_SAMPLES
                                    )
                                )

                                X = (
                                    anomaly_noise
                                    +
                                    signal[
                                        None,
                                        :
                                    ]
                                )

                                q_anomaly = (
                                    graph_energy_samples(
                                        X,
                                        L
                                    )
                                )

                                pd_mc = float(
                                    np.mean(
                                        q_anomaly
                                        >
                                        tau
                                    )
                                )

                                # ----------------------------
                                # Spectral H1
                                # ----------------------------

                                pd_theory = h1_sf(
                                    tau,
                                    signal,
                                    C,
                                    spectral_eigenvalues,
                                    spectral_eigenvectors
                                )

                                # ----------------------------
                                # Mean shift
                                # ----------------------------

                                theoretical_mean_shift = (
                                    graph_energy
                                )

                                empirical_mean_shift = float(
                                    np.mean(
                                        q_anomaly
                                    )
                                    -
                                    mean_q
                                )

                                rows.append(
                                    {
                                        "graph_type":
                                            graph_type,

                                        "n":
                                            n,

                                        "noise_model":
                                            noise_model,

                                        "sigma":
                                            sigma,

                                        "amplitude":
                                            amplitude,

                                        "pattern":
                                            pattern,

                                        "repetition":
                                            repetition,

                                        "signal_location":
                                            metadata[
                                                "signal_location"
                                            ],

                                        "signal_norm":
                                            signal_norm,

                                        "signal_graph_energy":
                                            graph_energy,

                                        "rayleigh_quotient":
                                            rayleigh,

                                        "lambda2":
                                            lambda2,

                                        "lambda_max":
                                            lambda_max,

                                        "normalized_rayleigh":
                                            (
                                                rayleigh
                                                /
                                                lambda_max
                                            ),

                                        "tau_spectral":
                                            tau,

                                        "pfa_mc":
                                            pfa_mc,

                                        "pfa_theory":
                                            pfa_theory,

                                        "pd_mc":
                                            pd_mc,

                                        "pd_theory":
                                            pd_theory,

                                        "pd_error":
                                            (
                                                pd_mc
                                                -
                                                pd_theory
                                            ),

                                        "theoretical_mean_shift":
                                            theoretical_mean_shift,

                                        "empirical_mean_shift":
                                            empirical_mean_shift,

                                        "mean_shift_error":
                                            (
                                                empirical_mean_shift
                                                -
                                                theoretical_mean_shift
                                            ),
                                    }
                                )

    # ========================================================
    # RAW
    # ========================================================

    raw = pd.DataFrame(
        rows
    )

    raw.to_csv(
        os.path.join(
            OUTPUT_DIR,
            "experiment9_raw.csv"
        ),
        index=False
    )

    # ========================================================
    # SUMMARY
    # ========================================================

    group_columns = [
        "graph_type",
        "n",
        "noise_model",
        "sigma",
        "amplitude",
        "pattern",
    ]

    summary = (
        raw
        .groupby(
            group_columns,
            as_index=False
        )
        .agg(
            signal_norm_mean=(
                "signal_norm",
                "mean"
            ),

            graph_energy_mean=(
                "signal_graph_energy",
                "mean"
            ),

            rayleigh_mean=(
                "rayleigh_quotient",
                "mean"
            ),

            normalized_rayleigh_mean=(
                "normalized_rayleigh",
                "mean"
            ),

            pfa_mc_mean=(
                "pfa_mc",
                "mean"
            ),

            pfa_theory_mean=(
                "pfa_theory",
                "mean"
            ),

            pd_mc_mean=(
                "pd_mc",
                "mean"
            ),

            pd_theory_mean=(
                "pd_theory",
                "mean"
            ),

            theoretical_mean_shift_mean=(
                "theoretical_mean_shift",
                "mean"
            ),

            empirical_mean_shift_mean=(
                "empirical_mean_shift",
                "mean"
            ),
        )
    )

    summary.to_csv(
        os.path.join(
            OUTPUT_DIR,
            "experiment9_summary.csv"
        ),
        index=False
    )

    # ========================================================
    # GLOBAL SPECTRAL ACCURACY
    # ========================================================

    global_metrics = regression_metrics(
        summary[
            "pd_mc_mean"
        ],
        summary[
            "pd_theory_mean"
        ]
    )

    metrics_df = pd.DataFrame(
        [
            {
                "method":
                    "full_spectral_arbitrary_signal",
                **global_metrics
            }
        ]
    )

    metrics_df.to_csv(
        os.path.join(
            OUTPUT_DIR,
            "experiment9_metrics.csv"
        ),
        index=False
    )

    # ========================================================
    # METRICS BY SIGNAL PATTERN
    # ========================================================

    pattern_rows = []

    for pattern in SIGNAL_PATTERNS:

        sub = summary[
            summary[
                "pattern"
            ]
            ==
            pattern
        ]

        metrics = regression_metrics(
            sub[
                "pd_mc_mean"
            ],
            sub[
                "pd_theory_mean"
            ]
        )

        pattern_rows.append(
            {
                "pattern":
                    pattern,

                "mean_PD_MC":
                    float(
                        sub[
                            "pd_mc_mean"
                        ].mean()
                    ),

                "mean_PD_theory":
                    float(
                        sub[
                            "pd_theory_mean"
                        ].mean()
                    ),

                "mean_graph_energy":
                    float(
                        sub[
                            "graph_energy_mean"
                        ].mean()
                    ),

                "mean_rayleigh":
                    float(
                        sub[
                            "rayleigh_mean"
                        ].mean()
                    ),

                **metrics
            }
        )

    pattern_metrics = pd.DataFrame(
        pattern_rows
    )

    pattern_metrics.to_csv(
        os.path.join(
            OUTPUT_DIR,
            "experiment9_metrics_by_pattern.csv"
        ),
        index=False
    )

    # ========================================================
    # METRICS BY NOISE MODEL
    # ========================================================

    noise_rows = []

    for noise_model in NOISE_MODELS:

        sub = summary[
            summary[
                "noise_model"
            ]
            ==
            noise_model
        ]

        metrics = regression_metrics(
            sub[
                "pd_mc_mean"
            ],
            sub[
                "pd_theory_mean"
            ]
        )

        noise_rows.append(
            {
                "noise_model":
                    noise_model,
                **metrics
            }
        )

    noise_metrics = pd.DataFrame(
        noise_rows
    )

    noise_metrics.to_csv(
        os.path.join(
            OUTPUT_DIR,
            "experiment9_metrics_by_noise.csv"
        ),
        index=False
    )

    # ========================================================
    # NULLSPACE / CONSTANT-SIGNAL CHECK
    # ========================================================

    constant = raw[
        raw[
            "pattern"
        ]
        ==
        "constant"
    ]

    nullspace_check = pd.DataFrame(
        [
            {
                "alpha":
                    ALPHA,

                "mean_constant_graph_energy":
                    float(
                        constant[
                            "signal_graph_energy"
                        ].mean()
                    ),

                "max_abs_constant_graph_energy":
                    float(
                        np.max(
                            np.abs(
                                constant[
                                    "signal_graph_energy"
                                ]
                            )
                        )
                    ),

                "mean_constant_PD_MC":
                    float(
                        constant[
                            "pd_mc"
                        ].mean()
                    ),

                "mean_constant_PD_theory":
                    float(
                        constant[
                            "pd_theory"
                        ].mean()
                    ),

                "mean_PFA_MC":
                    float(
                        constant[
                            "pfa_mc"
                        ].mean()
                    ),

                "mean_PFA_theory":
                    float(
                        constant[
                            "pfa_theory"
                        ].mean()
                    ),

                "mean_abs_PD_minus_alpha_MC":
                    float(
                        np.mean(
                            np.abs(
                                constant[
                                    "pd_mc"
                                ]
                                -
                                ALPHA
                            )
                        )
                    ),

                "max_abs_PD_theory_minus_alpha":
                    float(
                        np.max(
                            np.abs(
                                constant[
                                    "pd_theory"
                                ]
                                -
                                ALPHA
                            )
                        )
                    ),
            }
        ]
    )

    nullspace_check.to_csv(
        os.path.join(
            OUTPUT_DIR,
            "experiment9_nullspace_check.csv"
        ),
        index=False
    )

    # ========================================================
    # SAME-NORM SIGNAL GEOMETRY TABLE
    # ========================================================

    geometry = (
        summary
        .groupby(
            [
                "pattern",
                "amplitude"
            ],
            as_index=False
        )
        .agg(
            mean_graph_energy=(
                "graph_energy_mean",
                "mean"
            ),

            mean_rayleigh=(
                "rayleigh_mean",
                "mean"
            ),

            mean_PD_MC=(
                "pd_mc_mean",
                "mean"
            ),

            mean_PD_theory=(
                "pd_theory_mean",
                "mean"
            ),
        )
    )

    geometry.to_csv(
        os.path.join(
            OUTPUT_DIR,
            "experiment9_signal_geometry.csv"
        ),
        index=False
    )

    # ========================================================
    # PLOT: P_D VS GRAPH ENERGY
    # ========================================================

    plt.figure(
        figsize=(8, 6)
    )

    plt.scatter(
        summary[
            "graph_energy_mean"
        ],
        summary[
            "pd_mc_mean"
        ],
        alpha=0.60
    )

    plt.xlabel(
        r"Signal graph energy $s^T L s$"
    )

    plt.ylabel(
        r"Monte Carlo $P_D$"
    )

    plt.title(
        "Experiment 9: detectability vs graph energy"
    )

    plt.grid(
        alpha=0.25
    )

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            OUTPUT_DIR,
            "pd_vs_graph_energy.png"
        ),
        dpi=300
    )

    plt.close()

    # ========================================================
    # PLOT: THEORY VS MONTE CARLO
    # ========================================================

    plt.figure(
        figsize=(7, 7)
    )

    plt.scatter(
        summary[
            "pd_mc_mean"
        ],
        summary[
            "pd_theory_mean"
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
        "Spectral P_D"
    )

    plt.title(
        "Experiment 9: arbitrary signal validation"
    )

    plt.grid(
        alpha=0.25
    )

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            OUTPUT_DIR,
            "observed_vs_spectral.png"
        ),
        dpi=300
    )

    plt.close()

    # ========================================================
    # PRINT RESULTS
    # ========================================================

    print()
    print(
        "================================================"
    )
    print(
        "GLOBAL METRICS"
    )
    print(
        "================================================"
    )

    print(
        metrics_df.to_string(
            index=False
        )
    )

    print()
    print(
        "================================================"
    )
    print(
        "METRICS BY SIGNAL PATTERN"
    )
    print(
        "================================================"
    )

    print(
        pattern_metrics.to_string(
            index=False
        )
    )

    print()
    print(
        "================================================"
    )
    print(
        "NULLSPACE CHECK"
    )
    print(
        "================================================"
    )

    print(
        nullspace_check.to_string(
            index=False
        )
    )

    print()
    print(
        "================================================"
    )
    print(
        "EXPERIMENT 9 COMPLETED"
    )
    print(
        "================================================"
    )

    print(
        f"Results saved to: {OUTPUT_DIR}/"
    )


if __name__ == "__main__":
    main()
