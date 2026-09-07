import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx


# ============================================================
# EXPERIMENT 4
# OUT-OF-SAMPLE VALIDATION OF GESNR
# ============================================================


# ============================================================
# FROZEN MODEL PARAMETERS
# ============================================================

# ВАЖНО:
# Эти коэффициенты получены в Experiment 3.
# Здесь они НЕ переобучаются.

FROZEN_A = 2.560786
FROZEN_B = -2.730899


# ============================================================
# EXPERIMENT SETTINGS
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

REPETITIONS = 30

BACKGROUND_SAMPLES = 3000

NORMAL_TEST_SAMPLES = 3000

ANOMALY_TEST_SAMPLES = 3000

MASTER_SEED = 20260901

OUTPUT_DIR = "results/experiment_04"

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# ============================================================
# FROZEN DETECTION MODEL
# ============================================================

def frozen_detection_probability(gesnr):
    """
    Модель из Experiment 3.

    P_D(R) =
    1 / [1 + exp(-(a ln(R) + b))]

    Коэффициенты a и b НЕ обучаются.
    """

    gesnr = np.maximum(
        gesnr,
        1e-12
    )

    z = (
        FROZEN_A
        *
        np.log(gesnr)
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
    """
    Строит связный граф Erdős-Rényi.

    Вероятность ребра выбирается так,
    чтобы средняя степень была умеренной.
    """

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

    # Если за 100 попыток связность не получена,
    # соединяем компоненты вручную.

    components = list(
        nx.connected_components(G)
    )

    for comp1, comp2 in zip(
        components[:-1],
        components[1:]
    ):

        u = next(
            iter(comp1)
        )

        v = next(
            iter(comp2)
        )

        G.add_edge(
            u,
            v
        )

    return G


def build_validation_graph(
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

        # k должно быть чётным и < n

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

        m = min(
            2,
            n - 1
        )

        seed = int(
            rng.integers(
                0,
                2**32 - 1
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
# GRAPH MATRICES
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


def graph_characteristics(
    G,
    L
):

    degrees = np.array(
        [
            degree
            for _, degree
            in G.degree()
        ],
        dtype=float
    )

    eigenvalues = np.linalg.eigvalsh(
        L
    )

    eigenvalues = np.sort(
        eigenvalues
    )

    return {
        "num_edges":
            G.number_of_edges(),

        "mean_degree":
            float(
                np.mean(degrees)
            ),

        "min_degree":
            float(
                np.min(degrees)
            ),

        "max_degree":
            float(
                np.max(degrees)
            ),

        "trace_L":
            float(
                np.trace(L)
            ),

        "trace_L2":
            float(
                np.trace(
                    L @ L
                )
            ),

        "lambda_2":
            float(
                eigenvalues[1]
            ),

        "lambda_max":
            float(
                eigenvalues[-1]
            )
    }


# ============================================================
# GRAPH ENERGY
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
    """
    Одиночная локальная аномалия:

        s = A e_k

    k выбирается равновероятно.
    """

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
# ADAPTIVE THRESHOLD
# ============================================================

def empirical_threshold(
    background_energies,
    alpha
):

    return float(
        np.quantile(
            background_energies,
            1.0 - alpha
        )
    )


# ============================================================
# METRICS
# ============================================================

def false_alarm_rate(
    energies,
    tau
):

    return float(
        np.mean(
            energies > tau
        )
    )


def detection_probability(
    energies,
    tau
):

    return float(
        np.mean(
            energies > tau
        )
    )


# ============================================================
# GESNR
# ============================================================

def calculate_gesnr(
    amplitude,
    mean_degree,
    sigma,
    trace_L2
):
    """
    GESNR =

              A^2 d
    ---------------------------
    sigma^2 sqrt(2 tr(L^2))

    Здесь d — средняя степень вершины,
    поскольку положение аномалии заранее
    неизвестно и выбирается равномерно.
    """

    numerator = (
        amplitude ** 2
        *
        mean_degree
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

    return float(
        numerator
        /
        denominator
    )


# ============================================================
# SINGLE MONTE-CARLO REPETITION
# ============================================================

def run_single_repetition(
    graph_type,
    n,
    noise,
    amplitude,
    rng
):

    # --------------------------------------------------------
    # NEW GRAPH
    # --------------------------------------------------------

    G = build_validation_graph(
        graph_type,
        n,
        rng
    )

    L = laplacian_matrix(
        G
    )

    graph_info = graph_characteristics(
        G,
        L
    )

    # --------------------------------------------------------
    # BACKGROUND CALIBRATION
    # --------------------------------------------------------

    X_background = generate_background(
        rng=rng,
        n_samples=BACKGROUND_SAMPLES,
        n_sensors=n,
        sigma=noise
    )

    E_background = graph_energy_batch(
        X_background,
        L
    )

    tau = empirical_threshold(
        E_background,
        ALPHA
    )

    # --------------------------------------------------------
    # INDEPENDENT H0 SAMPLE
    # --------------------------------------------------------

    X_normal = generate_background(
        rng=rng,
        n_samples=NORMAL_TEST_SAMPLES,
        n_sensors=n,
        sigma=noise
    )

    E_normal = graph_energy_batch(
        X_normal,
        L
    )

    pfa = false_alarm_rate(
        E_normal,
        tau
    )

    # --------------------------------------------------------
    # INDEPENDENT H1 SAMPLE
    # --------------------------------------------------------

    X_anomaly, anomaly_nodes = (
        generate_anomalies(
            rng=rng,
            n_samples=ANOMALY_TEST_SAMPLES,
            n_sensors=n,
            sigma=noise,
            amplitude=amplitude
        )
    )

    E_anomaly = graph_energy_batch(
        X_anomaly,
        L
    )

    observed_pd = detection_probability(
        E_anomaly,
        tau
    )

    # --------------------------------------------------------
    # ACTUAL ANOMALY DEGREES
    # --------------------------------------------------------

    degrees = np.array(
        [
            degree
            for _, degree
            in G.degree()
        ],
        dtype=float
    )

    actual_anomaly_degree = float(
        np.mean(
            degrees[
                anomaly_nodes
            ]
        )
    )

    # --------------------------------------------------------
    # GESNR
    # --------------------------------------------------------

    # Для предсказания используем mean_degree графа,
    # а НЕ фактические anomaly_nodes.
    #
    # Это важно: модель не должна "подглядывать"
    # в тестовые аномалии.

    gesnr = calculate_gesnr(
        amplitude=amplitude,
        mean_degree=graph_info[
            "mean_degree"
        ],
        sigma=noise,
        trace_L2=graph_info[
            "trace_L2"
        ]
    )

    predicted_pd = (
        frozen_detection_probability(
            gesnr
        )
    )

    # --------------------------------------------------------
    # RESULT
    # --------------------------------------------------------

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
            graph_info[
                "num_edges"
            ],

        "mean_degree":
            graph_info[
                "mean_degree"
            ],

        "actual_anomaly_degree":
            actual_anomaly_degree,

        "min_degree":
            graph_info[
                "min_degree"
            ],

        "max_degree":
            graph_info[
                "max_degree"
            ],

        "trace_L":
            graph_info[
                "trace_L"
            ],

        "trace_L2":
            graph_info[
                "trace_L2"
            ],

        "lambda_2":
            graph_info[
                "lambda_2"
            ],

        "lambda_max":
            graph_info[
                "lambda_max"
            ],

        "tau":
            tau,

        "pfa":
            pfa,

        "observed_pd":
            observed_pd,

        "gesnr":
            gesnr,

        "predicted_pd":
            predicted_pd,

        "residual":
            observed_pd
            -
            predicted_pd
    }


# ============================================================
# CONFIDENCE INTERVAL
# ============================================================

def ci95(
    values
):

    values = np.asarray(
        values,
        dtype=float
    )

    mean = np.mean(
        values
    )

    if len(values) <= 1:
        return (
            mean,
            0.0
        )

    std = np.std(
        values,
        ddof=1
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
# R^2
# ============================================================

def r_squared(
    observed,
    predicted
):

    observed = np.asarray(
        observed
    )

    predicted = np.asarray(
        predicted
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
            np.mean(
                observed
            )
        ) ** 2
    )

    return (
        1.0
        -
        ss_res / ss_tot
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print(
        "======================================"
    )
    print(
        "EXPERIMENT 4 — OUT-OF-SAMPLE TEST"
    )
    print(
        "======================================"
    )

    print()
    print(
        f"Frozen a = {FROZEN_A}"
    )

    print(
        f"Frozen b = {FROZEN_B}"
    )

    # --------------------------------------------------------
    # NUMBER OF RUNS
    # --------------------------------------------------------

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
        f"Всего Monte-Carlo запусков: "
        f"{total_runs}"
    )

    master_rng = np.random.default_rng(
        MASTER_SEED
    )

    raw_results = []

    counter = 0

    # ========================================================
    # EXPERIMENT LOOP
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
                                graph_type=graph_type,
                                n=n,
                                noise=noise,
                                amplitude=amplitude,
                                rng=rng
                            )
                        )

                        result[
                            "repetition"
                        ] = repetition

                        raw_results.append(
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
        raw_results
    )

    raw_file = os.path.join(
        OUTPUT_DIR,
        "experiment4_raw.csv"
    )

    raw_df.to_csv(
        raw_file,
        index=False
    )

    # ========================================================
    # AGGREGATION
    # ========================================================

    group_columns = [
        "graph_type",
        "num_sensors",
        "noise",
        "amplitude"
    ]

    metrics = [
        "num_edges",
        "mean_degree",
        "actual_anomaly_degree",
        "trace_L",
        "trace_L2",
        "lambda_2",
        "lambda_max",
        "tau",
        "pfa",
        "gesnr",
        "observed_pd",
        "predicted_pd",
        "residual"
    ]

    summary_rows = []

    grouped = raw_df.groupby(
        group_columns
    )

    for keys, subset in grouped:

        row = dict(
            zip(
                group_columns,
                keys
            )
        )

        for metric in metrics:

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
        "experiment4_summary.csv"
    )

    summary.to_csv(
        summary_file,
        index=False
    )

    # ========================================================
    # OUT-OF-SAMPLE METRICS
    # ========================================================

    observed = (
        summary[
            "observed_pd_mean"
        ].to_numpy()
    )

    predicted = (
        summary[
            "predicted_pd_mean"
        ].to_numpy()
    )

    residuals = (
        observed
        -
        predicted
    )

    mae = np.mean(
        np.abs(
            residuals
        )
    )

    rmse = np.sqrt(
        np.mean(
            residuals ** 2
        )
    )

    r2 = r_squared(
        observed,
        predicted
    )

    mean_bias = np.mean(
        residuals
    )

    mean_pfa = summary[
        "pfa_mean"
    ].mean()

    min_pfa = summary[
        "pfa_mean"
    ].min()

    max_pfa = summary[
        "pfa_mean"
    ].max()

    print()
    print(
        "======================================"
    )
    print(
        "OUT-OF-SAMPLE RESULTS"
    )
    print(
        "======================================"
    )

    print(
        f"R^2 test = {r2:.6f}"
    )

    print(
        f"MAE test = {mae:.6f}"
    )

    print(
        f"RMSE test = {rmse:.6f}"
    )

    print(
        f"Mean bias = {mean_bias:.6f}"
    )

    print()

    print(
        f"Mean P_FA = {mean_pfa:.6f}"
    )

    print(
        f"Min P_FA  = {min_pfa:.6f}"
    )

    print(
        f"Max P_FA  = {max_pfa:.6f}"
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

        pred = part[
            "predicted_pd_mean"
        ].to_numpy()

        res = obs - pred

        topology_mae = (
            np.mean(
                np.abs(res)
            )
        )

        topology_rmse = (
            np.sqrt(
                np.mean(
                    res ** 2
                )
            )
        )

        topology_r2 = (
            r_squared(
                obs,
                pred
            )
        )

        topology_bias = (
            np.mean(res)
        )

        topology_rows.append(
            {
                "graph_type":
                    graph_type,

                "R2":
                    topology_r2,

                "MAE":
                    topology_mae,

                "RMSE":
                    topology_rmse,

                "bias":
                    topology_bias,

                "mean_pfa":
                    part[
                        "pfa_mean"
                    ].mean()
            }
        )

    topology_metrics = pd.DataFrame(
        topology_rows
    )

    topology_metrics_file = os.path.join(
        OUTPUT_DIR,
        "metrics_by_topology.csv"
    )

    topology_metrics.to_csv(
        topology_metrics_file,
        index=False
    )

    print()
    print(
        "РЕЗУЛЬТАТЫ ПО ТОПОЛОГИЯМ"
    )

    print(
        topology_metrics.to_string(
            index=False
        )
    )

    # ========================================================
    # SAVE GLOBAL METRICS
    # ========================================================

    global_metrics = pd.DataFrame(
        {
            "metric": [
                "frozen_a",
                "frozen_b",
                "R2_test",
                "MAE_test",
                "RMSE_test",
                "mean_bias",
                "mean_PFA",
                "min_PFA",
                "max_PFA"
            ],

            "value": [
                FROZEN_A,
                FROZEN_B,
                r2,
                mae,
                rmse,
                mean_bias,
                mean_pfa,
                min_pfa,
                max_pfa
            ]
        }
    )

    metrics_file = os.path.join(
        OUTPUT_DIR,
        "experiment4_metrics.csv"
    )

    global_metrics.to_csv(
        metrics_file,
        index=False
    )

    # ========================================================
    # PLOT 1
    # OBSERVED vs PREDICTED
    # ========================================================

    plt.figure(
        figsize=(7, 7)
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
                "predicted_pd_mean"
            ],
            part[
                "observed_pd_mean"
            ],
            alpha=0.7,
            label=graph_type
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
        f"Out-of-sample validation, "
        f"R²={r2:.3f}"
    )

    plt.legend()

    plt.grid(
        alpha=0.25
    )

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            OUTPUT_DIR,
            "observed_vs_predicted_test.png"
        ),
        dpi=300
    )

    plt.close()

    # ========================================================
    # PLOT 2
    # PD vs GESNR
    # ========================================================

    plt.figure(
        figsize=(9, 6)
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
                "gesnr_mean"
            ],
            part[
                "observed_pd_mean"
            ],
            alpha=0.7,
            label=graph_type
        )

    r_min = max(
        summary[
            "gesnr_mean"
        ].min(),
        1e-4
    )

    r_max = summary[
        "gesnr_mean"
    ].max()

    r_grid = np.logspace(
        np.log10(r_min),
        np.log10(r_max),
        500
    )

    pd_grid = (
        frozen_detection_probability(
            r_grid
        )
    )

    plt.plot(
        r_grid,
        pd_grid,
        linewidth=2.5,
        label="Frozen model"
    )

    plt.xscale(
        "log"
    )

    plt.xlabel(
        "GESNR"
    )

    plt.ylabel(
        "Detection probability"
    )

    plt.title(
        "Out-of-sample GESNR validation"
    )

    plt.legend()

    plt.grid(
        alpha=0.25
    )

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            OUTPUT_DIR,
            "pd_vs_gesnr_test.png"
        ),
        dpi=300
    )

    plt.close()

    # ========================================================
    # PLOT 3
    # RESIDUALS
    # ========================================================

    plt.figure(
        figsize=(9, 6)
    )

    for graph_type in GRAPH_TYPES:

        part = summary[
            summary[
                "graph_type"
            ]
            ==
            graph_type
        ]

        residual = (
            part[
                "observed_pd_mean"
            ]
            -
            part[
                "predicted_pd_mean"
            ]
        )

        plt.scatter(
            part[
                "gesnr_mean"
            ],
            residual,
            alpha=0.7,
            label=graph_type
        )

    plt.axhline(
        0,
        linestyle="--"
    )

    plt.xscale(
        "log"
    )

    plt.xlabel(
        "GESNR"
    )

    plt.ylabel(
        "Observed - predicted"
    )

    plt.title(
        "Out-of-sample residuals"
    )

    plt.legend()

    plt.grid(
        alpha=0.25
    )

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            OUTPUT_DIR,
            "residuals_test.png"
        ),
        dpi=300
    )

    plt.close()

    # ========================================================
    # PLOT 4
    # PFA VALIDATION
    # ========================================================

    plt.figure(
        figsize=(9, 6)
    )

    for graph_type in GRAPH_TYPES:

        part = summary[
            summary[
                "graph_type"
            ]
            ==
            graph_type
        ]

        grouped_noise = (
            part.groupby(
                "noise"
            )[
                "pfa_mean"
            ]
            .mean()
        )

        plt.plot(
            grouped_noise.index,
            grouped_noise.values,
            marker="o",
            label=graph_type
        )

    plt.axhline(
        ALPHA,
        linestyle="--",
        label="Target alpha"
    )

    plt.xlabel(
        "Noise standard deviation"
    )

    plt.ylabel(
        "False alarm probability"
    )

    plt.title(
        "False alarm control on unseen graphs"
    )

    plt.legend()

    plt.grid(
        alpha=0.25
    )

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            OUTPUT_DIR,
            "pfa_unseen_graphs.png"
        ),
        dpi=300
    )

    plt.close()

    # ========================================================
    # FINISH
    # ========================================================

    print()
    print(
        "======================================"
    )
    print(
        "EXPERIMENT 4 COMPLETED"
    )
    print(
        "======================================"
    )

    print()
    print(
        f"Results saved to: "
        f"{OUTPUT_DIR}/"
    )


if __name__ == "__main__":
    main()
