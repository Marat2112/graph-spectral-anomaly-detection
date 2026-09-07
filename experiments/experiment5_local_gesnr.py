import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx


# ============================================================
# EXPERIMENT 5
# LOCAL NODE-DEPENDENT GESNR
# ============================================================


# ============================================================
# FROZEN MODEL FROM EXPERIMENT 3
# ============================================================

FROZEN_A = 2.560786
FROZEN_B = -2.730899


# ============================================================
# SETTINGS
# ============================================================

GRAPH_TYPES = [
    "star",
    "erdos_renyi",
    "watts_strogatz",
    "barabasi_albert"
]

NUM_SENSOR_VALUES = [
    15,
    30,
    60
]

NOISE_LEVELS = [
    0.15,
    0.25,
    0.35
]

ANOMALY_AMPLITUDES = [
    0.75,
    1.25,
    1.75,
    2.50
]

ALPHA = 0.01

REPETITIONS = 20

BACKGROUND_SAMPLES = 3000
NORMAL_TEST_SAMPLES = 3000
ANOMALY_TEST_SAMPLES = 4000

MASTER_SEED = 20260905

OUTPUT_DIR = "results/experiment_05"

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# ============================================================
# FROZEN LOGISTIC MODEL
# ============================================================

def frozen_pd_model(r):
    """
    Frozen model from Experiment 3:

        P_D(R) =
        1 / (1 + exp(-(a ln R + b)))
    """

    r = np.maximum(
        r,
        1e-12
    )

    z = (
        FROZEN_A
        *
        np.log(r)
        +
        FROZEN_B
    )

    return (
        1.0
        /
        (
            1.0
            +
            np.exp(-z)
        )
    )


# ============================================================
# GRAPH GENERATION
# ============================================================

def make_connected_erdos_renyi(
    n,
    rng
):

    p = min(
        4.0 / (n - 1),
        1.0
    )

    for _ in range(100):

        seed = int(
            rng.integers(
                0,
                2**32 - 1
            )
        )

        G = nx.erdos_renyi_graph(
            n=n,
            p=p,
            seed=seed
        )

        if nx.is_connected(G):
            return G

    components = list(
        nx.connected_components(G)
    )

    for comp1, comp2 in zip(
        components[:-1],
        components[1:]
    ):

        u = next(iter(comp1))
        v = next(iter(comp2))

        G.add_edge(
            u,
            v
        )

    return G


def build_graph(
    graph_type,
    n,
    rng
):

    if graph_type == "star":

        G = nx.star_graph(
            n - 1
        )

    elif graph_type == "erdos_renyi":

        G = make_connected_erdos_renyi(
            n,
            rng
        )

    elif graph_type == "watts_strogatz":

        k = min(
            4,
            n - 1
        )

        if k % 2 == 1:
            k -= 1

        seed = int(
            rng.integers(
                0,
                2**32 - 1
            )
        )

        G = nx.connected_watts_strogatz_graph(
            n=n,
            k=k,
            p=0.20,
            tries=100,
            seed=seed
        )

    elif graph_type == "barabasi_albert":

        seed = int(
            rng.integers(
                0,
                2**32 - 1
            )
        )

        G = nx.barabasi_albert_graph(
            n=n,
            m=2,
            seed=seed
        )

    else:

        raise ValueError(
            f"Unknown graph type: {graph_type}"
        )

    return G


# ============================================================
# LAPLACIAN
# ============================================================

def laplacian_matrix(G):

    A = nx.to_numpy_array(
        G,
        dtype=float
    )

    degree_vector = A.sum(
        axis=1
    )

    D = np.diag(
        degree_vector
    )

    return (
        D - A
    )


# ============================================================
# ENERGY
# ============================================================

def graph_energy_batch(
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
# DATA GENERATION
# ============================================================

def generate_background(
    rng,
    n_samples,
    n_sensors,
    sigma
):

    return rng.normal(
        loc=0.0,
        scale=sigma,
        size=(
            n_samples,
            n_sensors
        )
    )


def generate_anomalies(
    rng,
    n_samples,
    n_sensors,
    sigma,
    amplitude
):

    X = rng.normal(
        loc=0.0,
        scale=sigma,
        size=(
            n_samples,
            n_sensors
        )
    )

    anomaly_nodes = rng.integers(
        low=0,
        high=n_sensors,
        size=n_samples
    )

    X[
        np.arange(n_samples),
        anomaly_nodes
    ] += amplitude

    return (
        X,
        anomaly_nodes
    )


# ============================================================
# THRESHOLD
# ============================================================

def empirical_threshold(
    energies,
    alpha
):

    return float(
        np.quantile(
            energies,
            1.0 - alpha
        )
    )


# ============================================================
# LOCAL GESNR
# ============================================================

def local_gesnr(
    amplitude,
    node_degrees,
    sigma,
    trace_L2
):
    """
    Для каждой аномалии:

                    A^2 d_k
        R_k = ------------------------
              sigma^2 sqrt(2 tr L^2)
    """

    numerator = (
        amplitude ** 2
        *
        node_degrees
    )

    denominator = (
        sigma ** 2
        *
        np.sqrt(
            2.0
            *
            trace_L2
        )
    )

    return (
        numerator
        /
        denominator
    )


# ============================================================
# OLD MEAN-DEGREE GESNR
# ============================================================

def mean_degree_gesnr(
    amplitude,
    mean_degree,
    sigma,
    trace_L2
):

    return (
        amplitude ** 2
        *
        mean_degree
        /
        (
            sigma ** 2
            *
            np.sqrt(
                2.0
                *
                trace_L2
            )
        )
    )


# ============================================================
# SINGLE REPETITION
# ============================================================

def run_single_repetition(
    graph_type,
    n,
    noise,
    amplitude,
    rng
):

    # --------------------------------------------------------
    # GRAPH
    # --------------------------------------------------------

    G = build_graph(
        graph_type,
        n,
        rng
    )

    L = laplacian_matrix(
        G
    )

    degrees = np.array(
        [
            degree
            for _, degree
            in G.degree()
        ],
        dtype=float
    )

    mean_degree = float(
        np.mean(degrees)
    )

    trace_L = float(
        np.trace(L)
    )

    trace_L2 = float(
        np.trace(
            L @ L
        )
    )

    eigenvalues = np.linalg.eigvalsh(
        L
    )

    eigenvalues = np.sort(
        eigenvalues
    )

    lambda_2 = float(
        eigenvalues[1]
    )

    lambda_max = float(
        eigenvalues[-1]
    )

    degree_std = float(
        np.std(
            degrees,
            ddof=0
        )
    )

    degree_cv = (
        degree_std
        /
        mean_degree
        if mean_degree > 0
        else 0.0
    )

    # --------------------------------------------------------
    # CALIBRATION
    # --------------------------------------------------------

    X_bg = generate_background(
        rng,
        BACKGROUND_SAMPLES,
        n,
        noise
    )

    E_bg = graph_energy_batch(
        X_bg,
        L
    )

    tau = empirical_threshold(
        E_bg,
        ALPHA
    )

    # --------------------------------------------------------
    # H0 VALIDATION
    # --------------------------------------------------------

    X_normal = generate_background(
        rng,
        NORMAL_TEST_SAMPLES,
        n,
        noise
    )

    E_normal = graph_energy_batch(
        X_normal,
        L
    )

    pfa = float(
        np.mean(
            E_normal > tau
        )
    )

    # --------------------------------------------------------
    # H1
    # --------------------------------------------------------

    X_anomaly, anomaly_nodes = (
        generate_anomalies(
            rng,
            ANOMALY_TEST_SAMPLES,
            n,
            noise,
            amplitude
        )
    )

    E_anomaly = graph_energy_batch(
        X_anomaly,
        L
    )

    detected = (
        E_anomaly > tau
    )

    observed_pd = float(
        np.mean(
            detected
        )
    )

    # ========================================================
    # LOCAL NODE DEGREES
    # ========================================================

    anomaly_degrees = degrees[
        anomaly_nodes
    ]

    # ========================================================
    # LOCAL GESNR FOR EVERY ANOMALY
    # ========================================================

    local_r = local_gesnr(
        amplitude=amplitude,
        node_degrees=anomaly_degrees,
        sigma=noise,
        trace_L2=trace_L2
    )

    # Каждая отдельная аномалия получает
    # собственное предсказание P_D.

    local_predictions = (
        frozen_pd_model(
            local_r
        )
    )

    # Затем усредняем вероятности.
    #
    # Это E[f(R_k)], а не f(E[R_k]).

    predicted_pd_local = float(
        np.mean(
            local_predictions
        )
    )

    # ========================================================
    # OLD MEAN-DEGREE MODEL
    # ========================================================

    r_mean = mean_degree_gesnr(
        amplitude=amplitude,
        mean_degree=mean_degree,
        sigma=noise,
        trace_L2=trace_L2
    )

    predicted_pd_mean = float(
        frozen_pd_model(
            r_mean
        )
    )

    # ========================================================
    # CONDITIONAL STAR STATISTICS
    # ========================================================

    center_pd = np.nan
    leaf_pd = np.nan

    center_pred = np.nan
    leaf_pred = np.nan

    if graph_type == "star":

        # В nx.star_graph(n-1)
        # вершина 0 — центр.

        center_mask = (
            anomaly_nodes == 0
        )

        leaf_mask = (
            anomaly_nodes != 0
        )

        if np.any(
            center_mask
        ):

            center_pd = float(
                np.mean(
                    detected[
                        center_mask
                    ]
                )
            )

            center_pred = float(
                np.mean(
                    local_predictions[
                        center_mask
                    ]
                )
            )

        if np.any(
            leaf_mask
        ):

            leaf_pd = float(
                np.mean(
                    detected[
                        leaf_mask
                    ]
                )
            )

            leaf_pred = float(
                np.mean(
                    local_predictions[
                        leaf_mask
                    ]
                )
            )

    # ========================================================
    # RESULT
    # ========================================================

    return {

        "graph_type":
            graph_type,

        "num_sensors":
            n,

        "noise":
            noise,

        "amplitude":
            amplitude,

        "num_edges":
            G.number_of_edges(),

        "mean_degree":
            mean_degree,

        "degree_std":
            degree_std,

        "degree_cv":
            degree_cv,

        "min_degree":
            float(
                np.min(degrees)
            ),

        "max_degree":
            float(
                np.max(degrees)
            ),

        "trace_L":
            trace_L,

        "trace_L2":
            trace_L2,

        "lambda_2":
            lambda_2,

        "lambda_max":
            lambda_max,

        "tau":
            tau,

        "pfa":
            pfa,

        "observed_pd":
            observed_pd,

        # OLD MODEL

        "gesnr_mean":
            r_mean,

        "predicted_pd_mean":
            predicted_pd_mean,

        # LOCAL MODEL

        "mean_local_gesnr":
            float(
                np.mean(
                    local_r
                )
            ),

        "median_local_gesnr":
            float(
                np.median(
                    local_r
                )
            ),

        "predicted_pd_local":
            predicted_pd_local,

        # ERRORS

        "residual_mean":
            observed_pd
            -
            predicted_pd_mean,

        "residual_local":
            observed_pd
            -
            predicted_pd_local,

        # STAR

        "star_center_observed_pd":
            center_pd,

        "star_center_predicted_pd":
            center_pred,

        "star_leaf_observed_pd":
            leaf_pd,

        "star_leaf_predicted_pd":
            leaf_pred
    }


# ============================================================
# CI95
# ============================================================

def ci95(values):

    values = np.asarray(
        values,
        dtype=float
    )

    # remove NaN

    values = values[
        ~np.isnan(values)
    ]

    if len(values) == 0:
        return (
            np.nan,
            np.nan
        )

    mean = float(
        np.mean(values)
    )

    if len(values) == 1:
        return (
            mean,
            0.0
        )

    std = float(
        np.std(
            values,
            ddof=1
        )
    )

    half_width = (
        1.96
        *
        std
        /
        np.sqrt(
            len(values)
        )
    )

    return (
        mean,
        half_width
    )


# ============================================================
# R2
# ============================================================

def r_squared(
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

    ss_res = np.sum(
        (
            observed
            -
            predicted
        ) ** 2
    )

    ss_tot = np.sum(
        (
            observed
            -
            np.mean(observed)
        ) ** 2
    )

    return (
        1.0
        -
        ss_res
        /
        ss_tot
    )


# ============================================================
# ERROR METRICS
# ============================================================

def prediction_metrics(
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

    residual = (
        observed
        -
        predicted
    )

    return {

        "R2":
            r_squared(
                observed,
                predicted
            ),

        "MAE":
            float(
                np.mean(
                    np.abs(
                        residual
                    )
                )
            ),

        "RMSE":
            float(
                np.sqrt(
                    np.mean(
                        residual ** 2
                    )
                )
            ),

        "bias":
            float(
                np.mean(
                    residual
                )
            )
    }


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print(
        "========================================="
    )
    print(
        "EXPERIMENT 5 — LOCAL NODE GESNR"
    )
    print(
        "========================================="
    )

    print()
    print(
        f"Frozen a = {FROZEN_A}"
    )

    print(
        f"Frozen b = {FROZEN_B}"
    )

    total_runs = (
        len(GRAPH_TYPES)
        *
        len(NUM_SENSOR_VALUES)
        *
        len(NOISE_LEVELS)
        *
        len(ANOMALY_AMPLITUDES)
        *
        REPETITIONS
    )

    print()
    print(
        f"Всего запусков: {total_runs}"
    )

    master_rng = np.random.default_rng(
        MASTER_SEED
    )

    rows = []

    counter = 0

    # ========================================================
    # MONTE CARLO
    # ========================================================

    for graph_type in GRAPH_TYPES:

        for n in NUM_SENSOR_VALUES:

            for noise in NOISE_LEVELS:

                for amplitude in ANOMALY_AMPLITUDES:

                    for repetition in range(
                        REPETITIONS
                    ):

                        seed = int(
                            master_rng.integers(
                                0,
                                2**32 - 1
                            )
                        )

                        rng = (
                            np.random.default_rng(
                                seed
                            )
                        )

                        result = (
                            run_single_repetition(
                                graph_type,
                                n,
                                noise,
                                amplitude,
                                rng
                            )
                        )

                        result[
                            "repetition"
                        ] = repetition

                        rows.append(
                            result
                        )

                        counter += 1

                    print(
                        f"{counter}/{total_runs}"
                        f" | {graph_type}"
                        f" | n={n}"
                        f" | sigma={noise}"
                        f" | A={amplitude}"
                    )

    # ========================================================
    # RAW DATA
    # ========================================================

    raw_df = pd.DataFrame(
        rows
    )

    raw_file = os.path.join(
        OUTPUT_DIR,
        "experiment5_raw.csv"
    )

    raw_df.to_csv(
        raw_file,
        index=False
    )

    # ========================================================
    # SUMMARY
    # ========================================================

    group_cols = [
        "graph_type",
        "num_sensors",
        "noise",
        "amplitude"
    ]

    metric_columns = [
        "num_edges",
        "mean_degree",
        "degree_std",
        "degree_cv",
        "min_degree",
        "max_degree",
        "trace_L",
        "trace_L2",
        "lambda_2",
        "lambda_max",
        "tau",
        "pfa",
        "observed_pd",
        "gesnr_mean",
        "predicted_pd_mean",
        "mean_local_gesnr",
        "median_local_gesnr",
        "predicted_pd_local",
        "residual_mean",
        "residual_local",
        "star_center_observed_pd",
        "star_center_predicted_pd",
        "star_leaf_observed_pd",
        "star_leaf_predicted_pd"
    ]

    summary_rows = []

    grouped = raw_df.groupby(
        group_cols
    )

    for keys, subset in grouped:

        row = dict(
            zip(
                group_cols,
                keys
            )
        )

        for metric in metric_columns:

            mean, ci = ci95(
                subset[
                    metric
                ].values
            )

            row[
                f"{metric}_mean"
            ] = mean

            row[
                f"{metric}_ci95"
            ] = ci

        summary_rows.append(
            row
        )

    summary = pd.DataFrame(
        summary_rows
    )

    summary_file = os.path.join(
        OUTPUT_DIR,
        "experiment5_summary.csv"
    )

    summary.to_csv(
        summary_file,
        index=False
    )

    # ========================================================
    # GLOBAL OLD MODEL METRICS
    # ========================================================

    observed = summary[
        "observed_pd_mean"
    ].to_numpy()

    pred_mean = summary[
        "predicted_pd_mean_mean"
    ].to_numpy()

    pred_local = summary[
        "predicted_pd_local_mean"
    ].to_numpy()

    old_metrics = prediction_metrics(
        observed,
        pred_mean
    )

    local_metrics = prediction_metrics(
        observed,
        pred_local
    )

    print()
    print(
        "========================================="
    )
    print(
        "GLOBAL COMPARISON"
    )
    print(
        "========================================="
    )

    print()
    print(
        "OLD MEAN-DEGREE MODEL"
    )

    for key, value in old_metrics.items():

        print(
            f"{key} = {value:.6f}"
        )

    print()
    print(
        "LOCAL NODE-DEGREE MODEL"
    )

    for key, value in local_metrics.items():

        print(
            f"{key} = {value:.6f}"
        )

    # ========================================================
    # IMPROVEMENT
    # ========================================================

    mae_improvement = (
        old_metrics["MAE"]
        -
        local_metrics["MAE"]
    )

    rmse_improvement = (
        old_metrics["RMSE"]
        -
        local_metrics["RMSE"]
    )

    r2_improvement = (
        local_metrics["R2"]
        -
        old_metrics["R2"]
    )

    print()
    print(
        "========================================="
    )
    print(
        "IMPROVEMENT"
    )
    print(
        "========================================="
    )

    print(
        f"Delta R2   = {r2_improvement:.6f}"
    )

    print(
        f"Delta MAE  = {mae_improvement:.6f}"
    )

    print(
        f"Delta RMSE = {rmse_improvement:.6f}"
    )

    # ========================================================
    # METRICS BY TOPOLOGY
    # ========================================================

    topology_rows = []

    for graph_type in GRAPH_TYPES:

        part = summary[
            summary[
                "graph_type"
            ]
            ==
            graph_type
        ]

        obs = part[
            "observed_pd_mean"
        ].to_numpy()

        old_pred = part[
            "predicted_pd_mean_mean"
        ].to_numpy()

        local_pred = part[
            "predicted_pd_local_mean"
        ].to_numpy()

        old_m = prediction_metrics(
            obs,
            old_pred
        )

        local_m = prediction_metrics(
            obs,
            local_pred
        )

        topology_rows.append(
            {
                "graph_type":
                    graph_type,

                "old_R2":
                    old_m["R2"],

                "local_R2":
                    local_m["R2"],

                "old_MAE":
                    old_m["MAE"],

                "local_MAE":
                    local_m["MAE"],

                "old_RMSE":
                    old_m["RMSE"],

                "local_RMSE":
                    local_m["RMSE"],

                "old_bias":
                    old_m["bias"],

                "local_bias":
                    local_m["bias"]
            }
        )

    topology_df = pd.DataFrame(
        topology_rows
    )

    topology_file = os.path.join(
        OUTPUT_DIR,
        "experiment5_metrics_by_topology.csv"
    )

    topology_df.to_csv(
        topology_file,
        index=False
    )

    print()
    print(
        "========================================="
    )
    print(
        "METRICS BY TOPOLOGY"
    )
    print(
        "========================================="
    )

    print(
        topology_df.to_string(
            index=False
        )
    )

    # ========================================================
    # STAR CENTER vs LEAVES
    # ========================================================

    star_summary = summary[
        summary[
            "graph_type"
        ]
        ==
        "star"
    ].copy()

    star_file = os.path.join(
        OUTPUT_DIR,
        "star_center_vs_leaves.csv"
    )

    star_summary.to_csv(
        star_file,
        index=False
    )

    # ========================================================
    # SAVE GLOBAL METRICS
    # ========================================================

    global_metrics = pd.DataFrame(
        {
            "metric": [
                "old_R2",
                "local_R2",
                "old_MAE",
                "local_MAE",
                "old_RMSE",
                "local_RMSE",
                "old_bias",
                "local_bias",
                "delta_R2",
                "delta_MAE",
                "delta_RMSE",
                "mean_PFA"
            ],

            "value": [
                old_metrics["R2"],
                local_metrics["R2"],
                old_metrics["MAE"],
                local_metrics["MAE"],
                old_metrics["RMSE"],
                local_metrics["RMSE"],
                old_metrics["bias"],
                local_metrics["bias"],
                r2_improvement,
                mae_improvement,
                rmse_improvement,
                summary[
                    "pfa_mean"
                ].mean()
            ]
        }
    )

    metrics_file = os.path.join(
        OUTPUT_DIR,
        "experiment5_metrics.csv"
    )

    global_metrics.to_csv(
        metrics_file,
        index=False
    )

    # ========================================================
    # PLOT 1
    # OLD vs LOCAL OBSERVED-PREDICTED
    # ========================================================

    plt.figure(
        figsize=(8, 8)
    )

    plt.scatter(
        pred_mean,
        observed,
        alpha=0.55,
        label="Mean-degree GESNR"
    )

    plt.scatter(
        pred_local,
        observed,
        alpha=0.55,
        label="Local-node GESNR"
    )

    plt.plot(
        [0, 1],
        [0, 1],
        linestyle="--"
    )

    plt.xlabel(
        "Predicted detection probability"
    )

    plt.ylabel(
        "Observed detection probability"
    )

    plt.title(
        "Mean-degree vs local-node GESNR"
    )

    plt.legend()

    plt.grid(
        alpha=0.25
    )

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            OUTPUT_DIR,
            "old_vs_local_prediction.png"
        ),
        dpi=300
    )

    plt.close()

    # ========================================================
    # PLOT 2
    # LOCAL MODEL BY TOPOLOGY
    # ========================================================

    plt.figure(
        figsize=(8, 8)
    )

    for graph_type in GRAPH_TYPES:

        part = summary[
            summary[
                "graph_type"
            ]
            ==
            graph_type
        ]

        plt.scatter(
            part[
                "predicted_pd_local_mean"
            ],
            part[
                "observed_pd_mean"
            ],
            alpha=0.70,
            label=graph_type
        )

    plt.plot(
        [0, 1],
        [0, 1],
        linestyle="--"
    )

    plt.xlabel(
        "Local-GESNR predicted P_D"
    )

    plt.ylabel(
        "Observed P_D"
    )

    plt.title(
        "Local-node GESNR validation"
    )

    plt.legend()

    plt.grid(
        alpha=0.25
    )

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            OUTPUT_DIR,
            "local_prediction_by_topology.png"
        ),
        dpi=300
    )

    plt.close()

    # ========================================================
    # PLOT 3
    # STAR CENTER vs LEAVES
    # ========================================================

    valid_star = star_summary.dropna(
        subset=[
            "star_center_observed_pd_mean",
            "star_leaf_observed_pd_mean"
        ]
    )

    plt.figure(
        figsize=(9, 6)
    )

    plt.scatter(
        valid_star[
            "star_center_predicted_pd_mean"
        ],
        valid_star[
            "star_center_observed_pd_mean"
        ],
        label="Star center",
        alpha=0.75
    )

    plt.scatter(
        valid_star[
            "star_leaf_predicted_pd_mean"
        ],
        valid_star[
            "star_leaf_observed_pd_mean"
        ],
        label="Star leaves",
        alpha=0.75
    )

    plt.plot(
        [0, 1],
        [0, 1],
        linestyle="--"
    )

    plt.xlabel(
        "Predicted P_D"
    )

    plt.ylabel(
        "Observed P_D"
    )

    plt.title(
        "Star graph: center vs leaves"
    )

    plt.legend()

    plt.grid(
        alpha=0.25
    )

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            OUTPUT_DIR,
            "star_center_vs_leaves.png"
        ),
        dpi=300
    )

    plt.close()

    # ========================================================
    # PLOT 4
    # ERROR vs DEGREE HETEROGENEITY
    # ========================================================

    summary[
        "local_abs_error"
    ] = np.abs(
        summary[
            "observed_pd_mean"
        ]
        -
        summary[
            "predicted_pd_local_mean"
        ]
    )

    summary[
        "old_abs_error"
    ] = np.abs(
        summary[
            "observed_pd_mean"
        ]
        -
        summary[
            "predicted_pd_mean_mean"
        ]
    )

    plt.figure(
        figsize=(9, 6)
    )

    plt.scatter(
        summary[
            "degree_cv_mean"
        ],
        summary[
            "old_abs_error"
        ],
        alpha=0.55,
        label="Mean-degree"
    )

    plt.scatter(
        summary[
            "degree_cv_mean"
        ],
        summary[
            "local_abs_error"
        ],
        alpha=0.55,
        label="Local-node"
    )

    plt.xlabel(
        "Coefficient of variation of node degree"
    )

    plt.ylabel(
        "Absolute prediction error"
    )

    plt.title(
        "Prediction error vs degree heterogeneity"
    )

    plt.legend()

    plt.grid(
        alpha=0.25
    )

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            OUTPUT_DIR,
            "error_vs_degree_heterogeneity.png"
        ),
        dpi=300
    )

    plt.close()

    # ========================================================
    # SAVE UPDATED SUMMARY AGAIN
    # ========================================================

    summary.to_csv(
        summary_file,
        index=False
    )

    # ========================================================
    # FINISH
    # ========================================================

    print()
    print(
        "========================================="
    )
    print(
        "EXPERIMENT 5 COMPLETED"
    )
    print(
        "========================================="
    )

    print()

    print(
        f"Results saved in: {OUTPUT_DIR}/"
    )


if __name__ == "__main__":
    main()
