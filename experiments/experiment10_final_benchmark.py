import os
from collections import deque

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx


# ============================================================
# EXPERIMENT 10
# FINAL DETECTOR BENCHMARK
#
# Detectors:
#
# 1. Laplacian:
#       Q_L = x^T L x
#
# 2. Euclidean energy:
#       Q_E = ||x||^2
#
# 3. Maximum absolute deviation:
#       Q_max = max_i |x_i|
#
# 4. Mahalanobis energy:
#       Q_M = x^T Sigma^{-1} x
#
# Fair comparison:
# every detector is calibrated independently to
#
#       P_FA = alpha
#
# using the SAME calibration background sample.
#
# Then P_FA is checked on an independent H0 sample.
#
# ============================================================


OUTPUT_DIR = "results/experiment_10"

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


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

DETECTOR_NAMES = [
    "laplacian",
    "euclidean",
    "max_abs",
    "mahalanobis",
]

REPETITIONS = 4

ALPHA = 0.01

CALIBRATION_SAMPLES = 10000
NORMAL_TEST_SAMPLES = 10000
ANOMALY_TEST_SAMPLES = 10000

CORRELATION_STRENGTH = 0.45

MASTER_SEED = 20260930

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

        radius = (
            1.5
            *
            np.sqrt(
                np.log(n)
                /
                n
            )
        )

        for _ in range(300):

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
            "Graph must be connected."
        )

    return G


# ============================================================
# GRAPH MATRICES
# ============================================================

def graph_matrices(
    G
):

    A = nx.to_numpy_array(
        G,
        dtype=float
    )

    degrees = np.sum(
        A,
        axis=1
    )

    D = np.diag(
        degrees
    )

    L = D - A

    return (
        A,
        L,
        degrees
    )


# ============================================================
# COVARIANCE MODEL
# ============================================================

def normalized_adjacency(
    A
):

    degrees = np.sum(
        A,
        axis=1
    )

    inv_sqrt = np.zeros_like(
        degrees,
        dtype=float
    )

    mask = degrees > 0

    inv_sqrt[
        mask
    ] = (
        1.0
        /
        np.sqrt(
            degrees[
                mask
            ]
        )
    )

    D_inv = np.diag(
        inv_sqrt
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
            f"Unknown noise model: "
            f"{noise_model}"
        )

    Sigma = (
        Sigma
        +
        Sigma.T
    ) / 2.0

    min_eigenvalue = np.min(
        np.linalg.eigvalsh(
            Sigma
        )
    )

    if min_eigenvalue <= 0:

        raise RuntimeError(
            "Covariance matrix is not "
            "positive definite."
        )

    return Sigma


# ============================================================
# SIGNAL HELPERS
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

    result = []

    while queue:

        u = queue.popleft()

        result.append(
            u
        )

        if (
            len(result)
            >=
            cluster_size
        ):
            break

        for v in G.neighbors(u):

            if v not in visited:

                visited.add(
                    v
                )

                queue.append(
                    v
                )

    return result[
        :cluster_size
    ]


# ============================================================
# CONSTRUCT SIGNAL
#
# Every signal has:
#
#       ||s||_2 = amplitude
#
# ============================================================

def construct_signal(
    pattern,
    amplitude,
    G,
    L,
    rng
):

    n = L.shape[0]

    (
        eigenvalues,
        eigenvectors
    ) = np.linalg.eigh(
        L
    )

    metadata = {}

    # --------------------------------------------------------
    # CONSTANT / NULLSPACE
    # --------------------------------------------------------

    if pattern == "constant":

        base = np.ones(
            n
        )

        metadata[
            "signal_location"
        ] = "all"

    # --------------------------------------------------------
    # FIEDLER
    # --------------------------------------------------------

    elif pattern == "fiedler":

        base = eigenvectors[
            :,
            1
        ]

        metadata[
            "signal_location"
        ] = "fiedler"

    # --------------------------------------------------------
    # CONNECTED CLUSTER
    # --------------------------------------------------------

    elif pattern == "cluster":

        start_node = int(
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
            start_node,
            cluster_size
        )

        base = np.zeros(
            n
        )

        base[
            nodes
        ] = 1.0

        metadata[
            "signal_location"
        ] = ";".join(
            map(
                str,
                nodes
            )
        )

    # --------------------------------------------------------
    # POINT
    # --------------------------------------------------------

    elif pattern == "point":

        node = int(
            rng.integers(
                0,
                n
            )
        )

        base = np.zeros(
            n
        )

        base[
            node
        ] = 1.0

        metadata[
            "signal_location"
        ] = str(
            node
        )

    # --------------------------------------------------------
    # EDGE BIPOLAR
    # --------------------------------------------------------

    elif pattern == "edge_bipolar":

        edges = list(
            G.edges()
        )

        index = int(
            rng.integers(
                0,
                len(edges)
            )
        )

        u, v = edges[
            index
        ]

        base = np.zeros(
            n
        )

        base[
            u
        ] = 1.0

        base[
            v
        ] = -1.0

        metadata[
            "signal_location"
        ] = f"{u}-{v}"

    # --------------------------------------------------------
    # HIGHEST GRAPH FREQUENCY
    # --------------------------------------------------------

    elif pattern == "high_frequency":

        base = eigenvectors[
            :,
            -1
        ]

        metadata[
            "signal_location"
        ] = "lambda_max"

    else:

        raise ValueError(
            f"Unknown pattern: {pattern}"
        )

    signal = normalize_signal(
        base,
        amplitude
    )

    lambda2 = float(
        eigenvalues[
            1
        ]
    )

    lambda_max = float(
        eigenvalues[
            -1
        ]
    )

    return (
        signal,
        metadata,
        lambda2,
        lambda_max
    )


# ============================================================
# DETECTOR STATISTICS
# ============================================================

def detector_statistics(
    X,
    L,
    precision
):

    # --------------------------------------------------------
    # 1. Laplacian detector
    # --------------------------------------------------------

    laplacian = np.einsum(
        "bi,ij,bj->b",
        X,
        L,
        X
    )

    # --------------------------------------------------------
    # 2. Euclidean energy
    # --------------------------------------------------------

    euclidean = np.einsum(
        "bi,bi->b",
        X,
        X
    )

    # --------------------------------------------------------
    # 3. Maximum absolute deviation
    # --------------------------------------------------------

    max_abs = np.max(
        np.abs(
            X
        ),
        axis=1
    )

    # --------------------------------------------------------
    # 4. Mahalanobis energy
    # --------------------------------------------------------

    mahalanobis = np.einsum(
        "bi,ij,bj->b",
        X,
        precision,
        X
    )

    return {
        "laplacian":
            laplacian,

        "euclidean":
            euclidean,

        "max_abs":
            max_abs,

        "mahalanobis":
            mahalanobis,
    }


# ============================================================
# SIGNAL DESCRIPTORS
# ============================================================

def signal_descriptors(
    signal,
    L,
    precision
):

    norm2 = float(
        signal.T
        @ signal
    )

    norm = np.sqrt(
        norm2
    )

    graph_energy = float(
        signal.T
        @ L
        @ signal
    )

    mahalanobis_energy = float(
        signal.T
        @ precision
        @ signal
    )

    max_amplitude = float(
        np.max(
            np.abs(
                signal
            )
        )
    )

    rayleigh = (
        graph_energy
        /
        norm2
    )

    return {
        "signal_norm":
            norm,

        "signal_euclidean_energy":
            norm2,

        "signal_graph_energy":
            graph_energy,

        "signal_mahalanobis_energy":
            mahalanobis_energy,

        "signal_max_abs":
            max_amplitude,

        "rayleigh_quotient":
            rayleigh,
    }


# ============================================================
# EMPIRICAL THRESHOLD
# ============================================================

def empirical_threshold(
    values
):

    return float(
        np.quantile(
            values,
            1.0 - ALPHA
        )
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print(
        "=================================================="
    )
    print(
        "EXPERIMENT 10"
    )
    print(
        "FINAL DETECTOR BENCHMARK"
    )
    print(
        "=================================================="
    )

    base_total = (
        len(
            GRAPH_TYPES
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

    h1_total = (
        base_total
        *
        len(
            SIGNAL_AMPLITUDES
        )
        *
        len(
            SIGNAL_PATTERNS
        )
    )

    print(
        f"Base configurations: {base_total}"
    )

    print(
        f"H1 signal conditions: {h1_total}"
    )

    print(
        f"Detectors: {len(DETECTOR_NAMES)}"
    )

    print()

    rows = []

    base_counter = 0

    # ========================================================
    # MAIN LOOP
    # ========================================================

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

                        print()
                        print(
                            f"BASE "
                            f"[{base_counter}/{base_total}] "
                            f"graph={graph_type}, "
                            f"n={n}, "
                            f"noise={noise_model}, "
                            f"sigma={sigma}, "
                            f"rep={repetition}"
                        )

                        # ====================================
                        # GRAPH
                        # ====================================

                        G = generate_graph(
                            graph_type,
                            n,
                            rng
                        )

                        (
                            A_graph,
                            L,
                            degrees
                        ) = graph_matrices(
                            G
                        )

                        # ====================================
                        # COVARIANCE
                        # ====================================

                        Sigma = build_covariance(
                            noise_model,
                            sigma,
                            A_graph
                        )

                        precision = np.linalg.inv(
                            Sigma
                        )

                        # ====================================
                        # GRAPH DESCRIPTORS
                        # ====================================

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

                        # ====================================
                        # CALIBRATION H0
                        # ====================================

                        calibration_noise = (
                            rng.multivariate_normal(
                                mean=np.zeros(n),
                                cov=Sigma,
                                size=CALIBRATION_SAMPLES
                            )
                        )

                        calibration_stats = (
                            detector_statistics(
                                calibration_noise,
                                L,
                                precision
                            )
                        )

                        thresholds = {}

                        for detector in (
                            DETECTOR_NAMES
                        ):

                            thresholds[
                                detector
                            ] = empirical_threshold(
                                calibration_stats[
                                    detector
                                ]
                            )

                        # ====================================
                        # INDEPENDENT H0 TEST
                        # ====================================

                        normal_test = (
                            rng.multivariate_normal(
                                mean=np.zeros(n),
                                cov=Sigma,
                                size=NORMAL_TEST_SAMPLES
                            )
                        )

                        normal_stats = (
                            detector_statistics(
                                normal_test,
                                L,
                                precision
                            )
                        )

                        pfa_values = {}

                        for detector in (
                            DETECTOR_NAMES
                        ):

                            pfa_values[
                                detector
                            ] = float(
                                np.mean(
                                    normal_stats[
                                        detector
                                    ]
                                    >
                                    thresholds[
                                        detector
                                    ]
                                )
                            )

                        # ====================================
                        # H1 CONDITIONS
                        # ====================================

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
                                    lambda2,
                                    lambda_max
                                ) = construct_signal(
                                    pattern,
                                    amplitude,
                                    G,
                                    L,
                                    rng
                                )

                                descriptors = (
                                    signal_descriptors(
                                        signal,
                                        L,
                                        precision
                                    )
                                )

                                # ----------------------------
                                # SAME anomaly sample used by
                                # ALL four detectors.
                                #
                                # This gives a paired and fair
                                # comparison.
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

                                anomaly_stats = (
                                    detector_statistics(
                                        X,
                                        L,
                                        precision
                                    )
                                )

                                for detector in (
                                    DETECTOR_NAMES
                                ):

                                    pd_value = float(
                                        np.mean(
                                            anomaly_stats[
                                                detector
                                            ]
                                            >
                                            thresholds[
                                                detector
                                            ]
                                        )
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

                                            "detector":
                                                detector,

                                            "signal_location":
                                                metadata[
                                                    "signal_location"
                                                ],

                                            "alpha":
                                                ALPHA,

                                            "threshold":
                                                thresholds[
                                                    detector
                                                ],

                                            "pfa":
                                                pfa_values[
                                                    detector
                                                ],

                                            "pd":
                                                pd_value,

                                            "mean_degree":
                                                mean_degree,

                                            "degree_cv":
                                                degree_cv,

                                            "trace_L":
                                                trace_L,

                                            "trace_L2":
                                                trace_L2,

                                            "lambda2":
                                                lambda2,

                                            "lambda_max":
                                                lambda_max,

                                            **descriptors,
                                        }
                                    )

    # ========================================================
    # RAW RESULTS
    # ========================================================

    raw = pd.DataFrame(
        rows
    )

    raw.to_csv(
        os.path.join(
            OUTPUT_DIR,
            "experiment10_raw.csv"
        ),
        index=False
    )

    # ========================================================
    # AGGREGATED SUMMARY
    # ========================================================

    condition_columns = [
        "graph_type",
        "n",
        "noise_model",
        "sigma",
        "amplitude",
        "pattern",
        "detector",
    ]

    summary = (
        raw
        .groupby(
            condition_columns,
            as_index=False
        )
        .agg(
            pfa_mean=(
                "pfa",
                "mean"
            ),

            pd_mean=(
                "pd",
                "mean"
            ),

            pd_std=(
                "pd",
                "std"
            ),

            threshold_mean=(
                "threshold",
                "mean"
            ),

            graph_energy_mean=(
                "signal_graph_energy",
                "mean"
            ),

            mahalanobis_energy_mean=(
                "signal_mahalanobis_energy",
                "mean"
            ),

            signal_max_abs_mean=(
                "signal_max_abs",
                "mean"
            ),

            rayleigh_mean=(
                "rayleigh_quotient",
                "mean"
            ),
        )
    )

    # ========================================================
    # RANK DETECTORS WITHIN EACH CONDITION
    # ========================================================

    ranking_group = [
        "graph_type",
        "n",
        "noise_model",
        "sigma",
        "amplitude",
        "pattern",
    ]

    summary[
        "pd_rank"
    ] = (
        summary
        .groupby(
            ranking_group
        )[
            "pd_mean"
        ]
        .rank(
            method="average",
            ascending=False
        )
    )

    summary[
        "is_winner"
    ] = (
        summary[
            "pd_rank"
        ]
        ==
        1.0
    )

    summary.to_csv(
        os.path.join(
            OUTPUT_DIR,
            "experiment10_summary.csv"
        ),
        index=False
    )

    # ========================================================
    # GLOBAL METHOD PERFORMANCE
    # ========================================================

    global_rows = []

    for detector in DETECTOR_NAMES:

        sub = summary[
            summary[
                "detector"
            ]
            ==
            detector
        ]

        global_rows.append(
            {
                "detector":
                    detector,

                "mean_PFA":
                    float(
                        sub[
                            "pfa_mean"
                        ].mean()
                    ),

                "mean_abs_PFA_error":
                    float(
                        np.mean(
                            np.abs(
                                sub[
                                    "pfa_mean"
                                ]
                                -
                                ALPHA
                            )
                        )
                    ),

                "mean_PD":
                    float(
                        sub[
                            "pd_mean"
                        ].mean()
                    ),

                "median_PD":
                    float(
                        sub[
                            "pd_mean"
                        ].median()
                    ),

                "mean_rank":
                    float(
                        sub[
                            "pd_rank"
                        ].mean()
                    ),

                "winner_count":
                    int(
                        sub[
                            "is_winner"
                        ].sum()
                    ),

                "winner_fraction":
                    float(
                        sub[
                            "is_winner"
                        ].mean()
                    ),
            }
        )

    global_metrics = pd.DataFrame(
        global_rows
    )

    global_metrics = (
        global_metrics
        .sort_values(
            "mean_rank"
        )
        .reset_index(
            drop=True
        )
    )

    global_metrics.to_csv(
        os.path.join(
            OUTPUT_DIR,
            "experiment10_global_metrics.csv"
        ),
        index=False
    )

    # ========================================================
    # METHOD x SIGNAL PATTERN
    # ========================================================

    pattern_table = (
        summary
        .groupby(
            [
                "pattern",
                "detector"
            ],
            as_index=False
        )
        .agg(
            mean_PFA=(
                "pfa_mean",
                "mean"
            ),

            mean_PD=(
                "pd_mean",
                "mean"
            ),

            median_PD=(
                "pd_mean",
                "median"
            ),

            mean_rank=(
                "pd_rank",
                "mean"
            ),

            winner_fraction=(
                "is_winner",
                "mean"
            ),

            mean_graph_energy=(
                "graph_energy_mean",
                "mean"
            ),
        )
    )

    pattern_table.to_csv(
        os.path.join(
            OUTPUT_DIR,
            "experiment10_by_pattern.csv"
        ),
        index=False
    )

    # ========================================================
    # METHOD x NOISE MODEL
    # ========================================================

    noise_table = (
        summary
        .groupby(
            [
                "noise_model",
                "detector"
            ],
            as_index=False
        )
        .agg(
            mean_PFA=(
                "pfa_mean",
                "mean"
            ),

            mean_PD=(
                "pd_mean",
                "mean"
            ),

            mean_rank=(
                "pd_rank",
                "mean"
            ),

            winner_fraction=(
                "is_winner",
                "mean"
            ),
        )
    )

    noise_table.to_csv(
        os.path.join(
            OUTPUT_DIR,
            "experiment10_by_noise.csv"
        ),
        index=False
    )

    # ========================================================
    # METHOD x GRAPH
    # ========================================================

    graph_table = (
        summary
        .groupby(
            [
                "graph_type",
                "detector"
            ],
            as_index=False
        )
        .agg(
            mean_PFA=(
                "pfa_mean",
                "mean"
            ),

            mean_PD=(
                "pd_mean",
                "mean"
            ),

            mean_rank=(
                "pd_rank",
                "mean"
            ),

            winner_fraction=(
                "is_winner",
                "mean"
            ),
        )
    )

    graph_table.to_csv(
        os.path.join(
            OUTPUT_DIR,
            "experiment10_by_graph.csv"
        ),
        index=False
    )

    # ========================================================
    # METHOD x AMPLITUDE
    # ========================================================

    amplitude_table = (
        summary
        .groupby(
            [
                "amplitude",
                "detector"
            ],
            as_index=False
        )
        .agg(
            mean_PFA=(
                "pfa_mean",
                "mean"
            ),

            mean_PD=(
                "pd_mean",
                "mean"
            ),

            mean_rank=(
                "pd_rank",
                "mean"
            ),
        )
    )

    amplitude_table.to_csv(
        os.path.join(
            OUTPUT_DIR,
            "experiment10_by_amplitude.csv"
        ),
        index=False
    )

    # ========================================================
    # NULLSPACE CHECK
    # ========================================================

    constant = summary[
        summary[
            "pattern"
        ]
        ==
        "constant"
    ]

    nullspace_table = (
        constant
        .groupby(
            [
                "amplitude",
                "detector"
            ],
            as_index=False
        )
        .agg(
            mean_PFA=(
                "pfa_mean",
                "mean"
            ),

            mean_PD=(
                "pd_mean",
                "mean"
            ),

            mean_rank=(
                "pd_rank",
                "mean"
            ),

            graph_energy=(
                "graph_energy_mean",
                "mean"
            ),
        )
    )

    nullspace_table.to_csv(
        os.path.join(
            OUTPUT_DIR,
            "experiment10_nullspace_comparison.csv"
        ),
        index=False
    )

    # ========================================================
    # HIGH-FREQUENCY CHECK
    # ========================================================

    high_frequency = summary[
        summary[
            "pattern"
        ]
        ==
        "high_frequency"
    ]

    high_frequency_table = (
        high_frequency
        .groupby(
            [
                "amplitude",
                "detector"
            ],
            as_index=False
        )
        .agg(
            mean_PFA=(
                "pfa_mean",
                "mean"
            ),

            mean_PD=(
                "pd_mean",
                "mean"
            ),

            mean_rank=(
                "pd_rank",
                "mean"
            ),

            winner_fraction=(
                "is_winner",
                "mean"
            ),
        )
    )

    high_frequency_table.to_csv(
        os.path.join(
            OUTPUT_DIR,
            "experiment10_high_frequency.csv"
        ),
        index=False
    )

    # ========================================================
    # POINT SIGNAL CHECK
    # ========================================================

    point = summary[
        summary[
            "pattern"
        ]
        ==
        "point"
    ]

    point_table = (
        point
        .groupby(
            [
                "amplitude",
                "detector"
            ],
            as_index=False
        )
        .agg(
            mean_PFA=(
                "pfa_mean",
                "mean"
            ),

            mean_PD=(
                "pd_mean",
                "mean"
            ),

            mean_rank=(
                "pd_rank",
                "mean"
            ),

            winner_fraction=(
                "is_winner",
                "mean"
            ),
        )
    )

    point_table.to_csv(
        os.path.join(
            OUTPUT_DIR,
            "experiment10_point_signal.csv"
        ),
        index=False
    )

    # ========================================================
    # BEST METHOD FOR EACH SIGNAL PATTERN
    # ========================================================

    average_pattern_pd = (
        pattern_table
        .copy()
    )

    best_indices = (
        average_pattern_pd
        .groupby(
            "pattern"
        )[
            "mean_PD"
        ]
        .idxmax()
    )

    best_by_pattern = (
        average_pattern_pd
        .loc[
            best_indices
        ]
        .sort_values(
            "pattern"
        )
        .reset_index(
            drop=True
        )
    )

    best_by_pattern.to_csv(
        os.path.join(
            OUTPUT_DIR,
            "experiment10_best_detector_by_pattern.csv"
        ),
        index=False
    )

    # ========================================================
    # PLOT 1
    # MEAN PD BY SIGNAL PATTERN
    # ========================================================

    pattern_order = SIGNAL_PATTERNS

    x = np.arange(
        len(
            pattern_order
        )
    )

    width = (
        0.8
        /
        len(
            DETECTOR_NAMES
        )
    )

    plt.figure(
        figsize=(12, 7)
    )

    for index, detector in enumerate(
        DETECTOR_NAMES
    ):

        values = []

        for pattern in pattern_order:

            row = pattern_table[
                (
                    pattern_table[
                        "pattern"
                    ]
                    ==
                    pattern
                )
                &
                (
                    pattern_table[
                        "detector"
                    ]
                    ==
                    detector
                )
            ]

            values.append(
                float(
                    row[
                        "mean_PD"
                    ].iloc[0]
                )
            )

        positions = (
            x
            +
            (
                index
                -
                (
                    len(
                        DETECTOR_NAMES
                    )
                    -
                    1
                )
                /
                2
            )
            *
            width
        )

        plt.bar(
            positions,
            values,
            width=width,
            label=detector
        )

    plt.xticks(
        x,
        pattern_order,
        rotation=25,
        ha="right"
    )

    plt.ylabel(
        "Mean detection probability"
    )

    plt.xlabel(
        "Signal geometry"
    )

    plt.title(
        "Experiment 10: detector performance by signal geometry"
    )

    plt.legend()

    plt.grid(
        axis="y",
        alpha=0.25
    )

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            OUTPUT_DIR,
            "detector_performance_by_pattern.png"
        ),
        dpi=300
    )

    plt.close()

    # ========================================================
    # PLOT 2
    # LAPLACIAN PD VS GRAPH ENERGY
    # ========================================================

    laplacian_summary = summary[
        summary[
            "detector"
        ]
        ==
        "laplacian"
    ]

    plt.figure(
        figsize=(8, 6)
    )

    plt.scatter(
        laplacian_summary[
            "graph_energy_mean"
        ],
        laplacian_summary[
            "pd_mean"
        ],
        alpha=0.60
    )

    plt.xlabel(
        r"Signal graph energy $s^T L s$"
    )

    plt.ylabel(
        r"Laplacian detector $P_D$"
    )

    plt.title(
        "Laplacian detectability vs graph energy"
    )

    plt.grid(
        alpha=0.25
    )

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            OUTPUT_DIR,
            "laplacian_pd_vs_graph_energy.png"
        ),
        dpi=300
    )

    plt.close()

    # ========================================================
    # PLOT 3
    # MEAN RANK
    # ========================================================

    rank_plot = (
        global_metrics
        .sort_values(
            "mean_rank"
        )
    )

    plt.figure(
        figsize=(8, 6)
    )

    plt.bar(
        rank_plot[
            "detector"
        ],
        rank_plot[
            "mean_rank"
        ]
    )

    plt.ylabel(
        "Mean rank (1 = best)"
    )

    plt.xlabel(
        "Detector"
    )

    plt.title(
        "Experiment 10: overall detector ranking"
    )

    plt.grid(
        axis="y",
        alpha=0.25
    )

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            OUTPUT_DIR,
            "overall_detector_rank.png"
        ),
        dpi=300
    )

    plt.close()

    # ========================================================
    # PRINT FINAL RESULTS
    # ========================================================

    print()
    print(
        "=================================================="
    )
    print(
        "GLOBAL BENCHMARK"
    )
    print(
        "=================================================="
    )

    print(
        global_metrics.to_string(
            index=False
        )
    )

    print()
    print(
        "=================================================="
    )
    print(
        "PERFORMANCE BY SIGNAL PATTERN"
    )
    print(
        "=================================================="
    )

    print(
        pattern_table.to_string(
            index=False
        )
    )

    print()
    print(
        "=================================================="
    )
    print(
        "BEST DETECTOR BY SIGNAL PATTERN"
    )
    print(
        "=================================================="
    )

    print(
        best_by_pattern.to_string(
            index=False
        )
    )

    print()
    print(
        "=================================================="
    )
    print(
        "NULLSPACE / CONSTANT SIGNAL"
    )
    print(
        "=================================================="
    )

    print(
        nullspace_table.to_string(
            index=False
        )
    )

    print()
    print(
        "=================================================="
    )
    print(
        "HIGH-FREQUENCY SIGNAL"
    )
    print(
        "=================================================="
    )

    print(
        high_frequency_table.to_string(
            index=False
        )
    )

    print()
    print(
        "=================================================="
    )
    print(
        "EXPERIMENT 10 COMPLETED"
    )
    print(
        "=================================================="
    )

    print(
        f"Results saved to: {OUTPUT_DIR}/"
    )


if __name__ == "__main__":
    main()
