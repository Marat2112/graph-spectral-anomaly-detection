import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from config import ExperimentConfig

from graph_model import (
    build_graph,
    laplacian_matrix,
    graph_energy_batch,
    graph_spectral_characteristics
)

from detector import (
    empirical_threshold,
    false_alarm_rate,
    detection_probability
)


# ============================================================
# SETTINGS
# ============================================================

GRAPH_TYPES = [
    "path",
    "ring",
    "grid",
    "random_geometric"
]

NUM_SENSOR_VALUES = [
    10,
    20,
    40,
    80
]

NOISE_LEVELS = [
    0.1,
    0.2,
    0.3
]

ANOMALY_AMPLITUDES = [
    0.5,
    1.0,
    1.5,
    2.0,
    3.0
]

REPETITIONS = 20

ALPHA = 0.01

BACKGROUND_SAMPLES = 2000

NORMAL_TEST_SAMPLES = 2000

ANOMALY_TEST_SAMPLES = 2000

SEED = 2026


# ============================================================
# DATA GENERATION
# ============================================================

def generate_background(
    rng,
    n_samples,
    n_sensors,
    sigma
):
    """
    H0:
        x = epsilon
    """

    return rng.normal(
        loc=0.0,
        scale=sigma,
        size=(
            n_samples,
            n_sensors
        )
    )


def generate_single_node_anomaly(
    rng,
    n_samples,
    n_sensors,
    sigma,
    amplitude
):
    """
    H1:
        x = s + epsilon

    Аномалия локализована
    в одной случайной вершине.
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

    return X, anomaly_nodes


# ============================================================
# CI
# ============================================================

def ci95(values):

    values = np.asarray(
        values,
        dtype=float
    )

    mean = np.mean(
        values
    )

    if len(values) <= 1:
        return mean, 0.0

    std = np.std(
        values,
        ddof=1
    )

    half_width = (
        1.96
        * std
        / np.sqrt(
            len(values)
        )
    )

    return mean, half_width


# ============================================================
# SINGLE EXPERIMENT
# ============================================================

def single_run(
    graph_type,
    num_sensors,
    noise,
    amplitude,
    rng
):

    # -----------------------------
    # GRAPH
    # -----------------------------

    G = build_graph(
        num_nodes=num_sensors,
        graph_type=graph_type,
        rng=rng
    )

    L = laplacian_matrix(
        G
    )

    spectral = graph_spectral_characteristics(
        L
    )

    degrees = np.array([
        d
        for _, d
        in G.degree()
    ])

    mean_degree = float(
        np.mean(
            degrees
        )
    )

    # -----------------------------
    # CALIBRATION H0
    # -----------------------------

    X_bg = generate_background(
        rng,
        BACKGROUND_SAMPLES,
        num_sensors,
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

    # -----------------------------
    # TEST H0
    # -----------------------------

    X_normal = generate_background(
        rng,
        NORMAL_TEST_SAMPLES,
        num_sensors,
        noise
    )

    E_normal = graph_energy_batch(
        X_normal,
        L
    )

    pfa = false_alarm_rate(
        E_normal,
        tau
    )

    # -----------------------------
    # TEST H1
    # -----------------------------

    X_anomaly, anomaly_nodes = (
        generate_single_node_anomaly(
            rng,
            ANOMALY_TEST_SAMPLES,
            num_sensors,
            noise,
            amplitude
        )
    )

    E_anomaly = graph_energy_batch(
        X_anomaly,
        L
    )

    pd = detection_probability(
        E_anomaly,
        tau
    )

    # -----------------------------
    # ANOMALY NODE DEGREE
    # -----------------------------

    anomaly_degrees = degrees[
        anomaly_nodes
    ]

    mean_anomaly_degree = float(
        np.mean(
            anomaly_degrees
        )
    )

    return {
        "graph_type":
            graph_type,

        "num_sensors":
            num_sensors,

        "noise":
            noise,

        "amplitude":
            amplitude,

        "num_edges":
            G.number_of_edges(),

        "mean_degree":
            mean_degree,

        "mean_anomaly_degree":
            mean_anomaly_degree,

        "trace_L":
            spectral[
                "trace_L"
            ],

        "trace_L2":
            spectral[
                "trace_L2"
            ],

        "lambda_2":
            spectral[
                "lambda_2"
            ],

        "lambda_max":
            spectral[
                "lambda_max"
            ],

        "tau":
            tau,

        "pfa":
            pfa,

        "pd":
            pd
    }


# ============================================================
# MAIN
# ============================================================

def main():

    results_dir = "results/experiment_02"

    os.makedirs(
        results_dir,
        exist_ok=True
    )

    rng_master = np.random.default_rng(
        SEED
    )

    results = []

    total = (
        len(GRAPH_TYPES)
        * len(NUM_SENSOR_VALUES)
        * len(NOISE_LEVELS)
        * len(ANOMALY_AMPLITUDES)
        * REPETITIONS
    )

    counter = 0

    print(
        "Начинаем эксперимент."
    )

    print(
        f"Всего запусков: {total}"
    )

    for graph_type in GRAPH_TYPES:

        for n in NUM_SENSOR_VALUES:

            for noise in NOISE_LEVELS:

                for amplitude in ANOMALY_AMPLITUDES:

                    for repetition in range(
                        REPETITIONS
                    ):

                        seed = rng_master.integers(
                            0,
                            2**32 - 1
                        )

                        rng = np.random.default_rng(
                            seed
                        )

                        result = single_run(
                            graph_type=graph_type,
                            num_sensors=n,
                            noise=noise,
                            amplitude=amplitude,
                            rng=rng
                        )

                        result[
                            "repetition"
                        ] = repetition

                        results.append(
                            result
                        )

                        counter += 1

                    print(
                        f"{counter}/{total} | "
                        f"{graph_type} | "
                        f"n={n} | "
                        f"sigma={noise} | "
                        f"A={amplitude}"
                    )

    # ========================================================
    # RAW CSV
    # ========================================================

    df = pd.DataFrame(
        results
    )

    raw_path = os.path.join(
        results_dir,
        "topology_raw.csv"
    )

    df.to_csv(
        raw_path,
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

    metrics = [
        "num_edges",
        "mean_degree",
        "mean_anomaly_degree",
        "trace_L",
        "trace_L2",
        "lambda_2",
        "lambda_max",
        "tau",
        "pfa",
        "pd"
    ]

    summary_rows = []

    grouped = df.groupby(
        group_cols
    )

    for keys, subset in grouped:

        row = dict(
            zip(
                group_cols,
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

    summary_path = os.path.join(
        results_dir,
        "topology_summary.csv"
    )

    summary.to_csv(
        summary_path,
        index=False
    )

    # ========================================================
    # PLOT 1
    # PD vs AMPLITUDE
    # ========================================================

    selected_n = 20
    selected_noise = 0.2

    subset = summary[
        (
            summary[
                "num_sensors"
            ] == selected_n
        )
        &
        (
            summary[
                "noise"
            ] == selected_noise
        )
    ]

    plt.figure(
        figsize=(8, 5)
    )

    for graph_type in GRAPH_TYPES:

        part = subset[
            subset[
                "graph_type"
            ] == graph_type
        ].sort_values(
            "amplitude"
        )

        plt.errorbar(
            part[
                "amplitude"
            ],
            part[
                "pd_mean"
            ],
            yerr=part[
                "pd_ci95"
            ],
            marker="o",
            label=graph_type
        )

    plt.xlabel(
        "Anomaly amplitude"
    )

    plt.ylabel(
        "Detection probability"
    )

    plt.title(
        f"P_D vs anomaly amplitude "
        f"(n={selected_n}, sigma={selected_noise})"
    )

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            results_dir,
            "pd_vs_amplitude_topology.png"
        ),
        dpi=300
    )

    plt.close()

    # ========================================================
    # PLOT 2
    # TAU vs TRACE L
    # ========================================================

    tau_plot = summary[
        (
            summary[
                "noise"
            ] == 0.2
        )
        &
        (
            summary[
                "amplitude"
            ] == 1.5
        )
    ]

    plt.figure(
        figsize=(8, 5)
    )

    for graph_type in GRAPH_TYPES:

        part = tau_plot[
            tau_plot[
                "graph_type"
            ] == graph_type
        ]

        plt.scatter(
            part[
                "trace_L_mean"
            ],
            part[
                "tau_mean"
            ],
            label=graph_type
        )

    plt.xlabel(
        "trace(L)"
    )

    plt.ylabel(
        "Adaptive threshold tau"
    )

    plt.title(
        "Adaptive threshold vs trace(L)"
    )

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            results_dir,
            "tau_vs_traceL.png"
        ),
        dpi=300
    )

    plt.close()

    # ========================================================
    # PLOT 3
    # PD vs MEAN DEGREE
    # ========================================================

    pd_degree = summary[
        (
            summary[
                "num_sensors"
            ] == 20
        )
        &
        (
            summary[
                "noise"
            ] == 0.2
        )
        &
        (
            summary[
                "amplitude"
            ] == 1.5
        )
    ]

    plt.figure(
        figsize=(8, 5)
    )

    plt.scatter(
        pd_degree[
            "mean_degree_mean"
        ],
        pd_degree[
            "pd_mean"
        ]
    )

    for _, row in pd_degree.iterrows():

        plt.annotate(
            row[
                "graph_type"
            ],
            (
                row[
                    "mean_degree_mean"
                ],
                row[
                    "pd_mean"
                ]
            )
        )

    plt.xlabel(
        "Mean node degree"
    )

    plt.ylabel(
        "Detection probability"
    )

    plt.title(
        "Detection probability vs graph connectivity"
    )

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            results_dir,
            "pd_vs_mean_degree.png"
        ),
        dpi=300
    )

    plt.close()

    # ========================================================
    # PLOT 4
    # PFA CONTROL
    # ========================================================

    pfa_plot = summary[
        (
            summary[
                "num_sensors"
            ] == 20
        )
        &
        (
            summary[
                "amplitude"
            ] == 1.5
        )
    ]

    plt.figure(
        figsize=(8, 5)
    )

    for graph_type in GRAPH_TYPES:

        part = pfa_plot[
            pfa_plot[
                "graph_type"
            ] == graph_type
        ].sort_values(
            "noise"
        )

        plt.plot(
            part[
                "noise"
            ],
            part[
                "pfa_mean"
            ],
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
        "False alarm control across graph topologies"
    )

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            results_dir,
            "pfa_topology.png"
        ),
        dpi=300
    )

    plt.close()

    print()
    print(
        "Эксперимент завершён."
    )

    print(
        f"Сырые результаты: {raw_path}"
    )

    print(
        f"Сводные результаты: {summary_path}"
    )


if __name__ == "__main__":
    main()
