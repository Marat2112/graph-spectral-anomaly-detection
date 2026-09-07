import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx

from scipy.optimize import minimize


# ============================================================
# EXPERIMENT 6
# STRICT TRAIN / VALIDATION COMPARISON
#
# Three criteria:
#
# M1: mean-degree GESNR
# M2: local-degree GESNR
# M3: full local variance-normalized criterion
#
# ============================================================


# ============================================================
# GENERAL SETTINGS
# ============================================================

ALPHA = 0.01

TRAIN_GRAPH_TYPES = [
    "path",
    "ring",
    "grid",
    "erdos_renyi",
    "watts_strogatz",
    "barabasi_albert",
]

VALID_GRAPH_TYPES = [
    "star",
    "random_geometric",
    "random_regular",
]


TRAIN_N = [
    12,
    20,
    36,
    50,
]

VALID_N = [
    15,
    30,
    60,
]


TRAIN_NOISE = [
    0.10,
    0.20,
    0.30,
]

VALID_NOISE = [
    0.15,
    0.25,
    0.35,
]


TRAIN_AMPLITUDE = [
    0.75,
    1.25,
    1.75,
    2.50,
]

VALID_AMPLITUDE = [
    1.00,
    1.50,
    2.00,
    3.00,
]


TRAIN_REPETITIONS = 10
VALID_REPETITIONS = 15


BACKGROUND_SAMPLES = 2000
NORMAL_SAMPLES = 2000
ANOMALY_SAMPLES = 3000


MASTER_SEED = 20260910


OUTPUT_DIR = "results/experiment_06"

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# ============================================================
# GRAPH HELPERS
# ============================================================

def ensure_connected(G):
    """
    Если граф несвязный, соединяем компоненты
    минимальным количеством рёбер.
    """

    if nx.is_connected(G):
        return G

    components = [
        list(component)
        for component
        in nx.connected_components(G)
    ]

    for c1, c2 in zip(
        components[:-1],
        components[1:]
    ):
        G.add_edge(
            c1[0],
            c2[0]
        )

    return G


# ============================================================
# GRID GRAPH
# ============================================================

def make_grid_graph(n):

    side = int(
        np.ceil(
            np.sqrt(n)
        )
    )

    G_full = nx.grid_2d_graph(
        side,
        side
    )

    nodes = list(
        G_full.nodes()
    )[:n]

    G = G_full.subgraph(
        nodes
    ).copy()

    G = nx.convert_node_labels_to_integers(
        G
    )

    G = ensure_connected(
        G
    )

    return G


# ============================================================
# ERDOS-RENYI
# ============================================================

def make_erdos_renyi(
    n,
    rng
):

    p = min(
        5.0 / (n - 1),
        0.90
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

    return ensure_connected(
        G
    )


# ============================================================
# RANDOM GEOMETRIC
# ============================================================

def make_random_geometric(
    n,
    rng
):

    radius = np.sqrt(
        8.0
        *
        np.log(n)
        /
        (
            np.pi
            *
            n
        )
    )

    radius = float(
        np.clip(
            radius,
            0.22,
            0.65
        )
    )

    for _ in range(100):

        seed = int(
            rng.integers(
                0,
                2**32 - 1
            )
        )

        G = nx.random_geometric_graph(
            n=n,
            radius=radius,
            seed=seed
        )

        if nx.is_connected(G):
            return G

    return ensure_connected(
        G
    )


# ============================================================
# GRAPH BUILDER
# ============================================================

def build_graph(
    graph_type,
    n,
    rng
):

    if graph_type == "path":

        return nx.path_graph(
            n
        )

    if graph_type == "ring":

        return nx.cycle_graph(
            n
        )

    if graph_type == "grid":

        return make_grid_graph(
            n
        )

    if graph_type == "erdos_renyi":

        return make_erdos_renyi(
            n,
            rng
        )

    if graph_type == "watts_strogatz":

        k = 4

        if k >= n:
            k = n - 1

        if k % 2 == 1:
            k -= 1

        seed = int(
            rng.integers(
                0,
                2**32 - 1
            )
        )

        return nx.connected_watts_strogatz_graph(
            n=n,
            k=k,
            p=0.20,
            tries=100,
            seed=seed
        )

    if graph_type == "barabasi_albert":

        seed = int(
            rng.integers(
                0,
                2**32 - 1
            )
        )

        return nx.barabasi_albert_graph(
            n=n,
            m=2,
            seed=seed
        )

    if graph_type == "star":

        return nx.star_graph(
            n - 1
        )

    if graph_type == "random_geometric":

        return make_random_geometric(
            n,
            rng
        )

    if graph_type == "random_regular":

        degree = 4

        seed = int(
            rng.integers(
                0,
                2**32 - 1
            )
        )

        G = nx.random_regular_graph(
            d=degree,
            n=n,
            seed=seed
        )

        return G

    raise ValueError(
        f"Unknown graph type: {graph_type}"
    )


# ============================================================
# LAPLACIAN MATRIX
# ============================================================

def laplacian_matrix(G):

    A = nx.to_numpy_array(
        G,
        dtype=float
    )

    degrees = A.sum(
        axis=1
    )

    D = np.diag(
        degrees
    )

    return (
        D - A
    )


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
# SIGNAL GENERATION
# ============================================================

def generate_background(
    rng,
    samples,
    n,
    sigma
):

    return rng.normal(
        loc=0.0,
        scale=sigma,
        size=(
            samples,
            n
        )
    )


def generate_anomalies(
    rng,
    samples,
    n,
    sigma,
    amplitude
):

    X = rng.normal(
        loc=0.0,
        scale=sigma,
        size=(
            samples,
            n
        )
    )

    nodes = rng.integers(
        0,
        n,
        size=samples
    )

    X[
        np.arange(samples),
        nodes
    ] += amplitude

    return (
        X,
        nodes
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
# THREE CRITERIA
# ============================================================

def criterion_mean(
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


def criterion_local(
    amplitude,
    degree,
    sigma,
    trace_L2
):

    return (
        amplitude ** 2
        *
        degree
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


def criterion_full(
    amplitude,
    degree,
    sigma,
    trace_L2
):

    numerator = (
        amplitude ** 2
        *
        degree
    )

    var_h1 = (
        2.0
        *
        sigma ** 4
        *
        trace_L2
        +
        4.0
        *
        amplitude ** 2
        *
        sigma ** 2
        *
        (
            degree ** 2
            +
            degree
        )
    )

    denominator = np.sqrt(
        var_h1
    )

    return (
        numerator
        /
        denominator
    )


# ============================================================
# ONE MONTE-CARLO GRAPH INSTANCE
# ============================================================

def simulate_instance(
    split,
    graph_type,
    n,
    sigma,
    amplitude,
    repetition,
    rng
):

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
        np.mean(
            degrees
        )
    )

    trace_L = float(
        np.trace(
            L
        )
    )

    trace_L2 = float(
        np.trace(
            L @ L
        )
    )

    eigenvalues = np.sort(
        np.linalg.eigvalsh(
            L
        )
    )

    lambda_2 = float(
        eigenvalues[1]
    )

    lambda_max = float(
        eigenvalues[-1]
    )

    degree_std = float(
        np.std(
            degrees
        )
    )

    degree_cv = (
        degree_std
        /
        mean_degree
    )

    # --------------------------------------------------------
    # THRESHOLD CALIBRATION
    # --------------------------------------------------------

    X_bg = generate_background(
        rng,
        BACKGROUND_SAMPLES,
        n,
        sigma
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
    # FALSE ALARM VALIDATION
    # --------------------------------------------------------

    X_h0 = generate_background(
        rng,
        NORMAL_SAMPLES,
        n,
        sigma
    )

    E_h0 = graph_energy_batch(
        X_h0,
        L
    )

    pfa = float(
        np.mean(
            E_h0 > tau
        )
    )

    # --------------------------------------------------------
    # H1
    # --------------------------------------------------------

    X_h1, anomaly_nodes = generate_anomalies(
        rng,
        ANOMALY_SAMPLES,
        n,
        sigma,
        amplitude
    )

    E_h1 = graph_energy_batch(
        X_h1,
        L
    )

    detected = (
        E_h1 > tau
    )

    # ========================================================
    # GROUP BY ACTUAL NODE DEGREE
    # ========================================================

    anomaly_degrees = degrees[
        anomaly_nodes
    ]

    unique_degrees = np.unique(
        anomaly_degrees
    )

    rows = []

    for degree in unique_degrees:

        mask = (
            anomaly_degrees
            ==
            degree
        )

        trials = int(
            np.sum(
                mask
            )
        )

        successes = int(
            np.sum(
                detected[
                    mask
                ]
            )
        )

        if trials == 0:
            continue

        r_mean = criterion_mean(
            amplitude,
            mean_degree,
            sigma,
            trace_L2
        )

        r_local = criterion_local(
            amplitude,
            degree,
            sigma,
            trace_L2
        )

        r_full = criterion_full(
            amplitude,
            degree,
            sigma,
            trace_L2
        )

        rows.append(
            {
                "split":
                    split,

                "graph_type":
                    graph_type,

                "num_sensors":
                    n,

                "noise":
                    sigma,

                "amplitude":
                    amplitude,

                "repetition":
                    repetition,

                "node_degree":
                    degree,

                "trials":
                    trials,

                "successes":
                    successes,

                "observed_pd":
                    successes
                    /
                    trials,

                "num_edges":
                    G.number_of_edges(),

                "mean_degree":
                    mean_degree,

                "degree_std":
                    degree_std,

                "degree_cv":
                    degree_cv,

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

                "R_mean":
                    r_mean,

                "R_local":
                    r_local,

                "R_full":
                    r_full,
            }
        )

    return rows


# ============================================================
# GENERATE DATASET
# ============================================================

def generate_dataset(
    split,
    graph_types,
    n_values,
    noise_values,
    amplitude_values,
    repetitions,
    seed
):

    master_rng = np.random.default_rng(
        seed
    )

    rows = []

    total = (
        len(graph_types)
        *
        len(n_values)
        *
        len(noise_values)
        *
        len(amplitude_values)
        *
        repetitions
    )

    counter = 0

    print()
    print(
        f"Generating {split} dataset"
    )

    print(
        f"Graph instances: {total}"
    )

    for graph_type in graph_types:

        for n in n_values:

            for sigma in noise_values:

                for amplitude in amplitude_values:

                    for repetition in range(
                        repetitions
                    ):

                        seed_i = int(
                            master_rng.integers(
                                0,
                                2**32 - 1
                            )
                        )

                        rng = (
                            np.random.default_rng(
                                seed_i
                            )
                        )

                        new_rows = simulate_instance(
                            split=split,
                            graph_type=graph_type,
                            n=n,
                            sigma=sigma,
                            amplitude=amplitude,
                            repetition=repetition,
                            rng=rng
                        )

                        rows.extend(
                            new_rows
                        )

                        counter += 1

                    print(
                        f"{counter}/{total}"
                        f" | {graph_type}"
                        f" | n={n}"
                        f" | sigma={sigma}"
                        f" | A={amplitude}"
                    )

    return pd.DataFrame(
        rows
    )


# ============================================================
# LOGISTIC MODEL
# ============================================================

def logistic_probability(
    R,
    a,
    b
):

    R = np.maximum(
        np.asarray(
            R,
            dtype=float
        ),
        1e-12
    )

    z = (
        a
        *
        np.log(R)
        +
        b
    )

    # numerical stability

    z = np.clip(
        z,
        -50,
        50
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
# BINOMIAL NEGATIVE LOG-LIKELIHOOD
# ============================================================

def binomial_nll(
    parameters,
    R,
    successes,
    trials
):

    a, b = parameters

    p = logistic_probability(
        R,
        a,
        b
    )

    eps = 1e-12

    p = np.clip(
        p,
        eps,
        1.0 - eps
    )

    failures = (
        trials
        -
        successes
    )

    nll = -np.sum(
        successes
        *
        np.log(p)
        +
        failures
        *
        np.log(
            1.0 - p
        )
    )

    return float(
        nll
    )


# ============================================================
# FIT ONE MODEL
# ============================================================

def fit_model(
    train_df,
    feature_column
):

    R = train_df[
        feature_column
    ].to_numpy(
        dtype=float
    )

    successes = train_df[
        "successes"
    ].to_numpy(
        dtype=float
    )

    trials = train_df[
        "trials"
    ].to_numpy(
        dtype=float
    )

    initial = np.array(
        [
            2.0,
            -2.0
        ]
    )

    result = minimize(
        binomial_nll,
        initial,
        args=(
            R,
            successes,
            trials
        ),
        method="BFGS"
    )

    if not result.success:

        print()
        print(
            f"WARNING for {feature_column}:"
        )

        print(
            result.message
        )

    a_fit = float(
        result.x[0]
    )

    b_fit = float(
        result.x[1]
    )

    nll = binomial_nll(
        result.x,
        R,
        successes,
        trials
    )

    total_trials = float(
        np.sum(
            trials
        )
    )

    nll_per_trial = (
        nll
        /
        total_trials
    )

    return {
        "criterion":
            feature_column,

        "a":
            a_fit,

        "b":
            b_fit,

        "train_nll":
            nll,

        "train_nll_per_trial":
            nll_per_trial,

        "optimizer_success":
            bool(
                result.success
            )
    }


# ============================================================
# ADD PREDICTIONS
# ============================================================

def apply_model(
    df,
    feature_column,
    a,
    b,
    prediction_column
):

    df[
        prediction_column
    ] = logistic_probability(
        df[
            feature_column
        ],
        a,
        b
    )


# ============================================================
# CONDITION-LEVEL AGGREGATION
# ============================================================

def aggregate_predictions(
    df,
    prediction_columns
):

    condition_columns = [
        "graph_type",
        "num_sensors",
        "noise",
        "amplitude",
    ]

    rows = []

    grouped = df.groupby(
        condition_columns
    )

    for keys, subset in grouped:

        total_trials = float(
            subset[
                "trials"
            ].sum()
        )

        total_success = float(
            subset[
                "successes"
            ].sum()
        )

        observed = (
            total_success
            /
            total_trials
        )

        row = dict(
            zip(
                condition_columns,
                keys
            )
        )

        row[
            "observed_pd"
        ] = observed

        row[
            "mean_pfa"
        ] = float(
            subset[
                "pfa"
            ].mean()
        )

        row[
            "degree_cv"
        ] = float(
            subset[
                "degree_cv"
            ].mean()
        )

        for column in prediction_columns:

            weighted_prediction = (
                np.sum(
                    subset[
                        column
                    ]
                    *
                    subset[
                        "trials"
                    ]
                )
                /
                total_trials
            )

            row[
                column
            ] = float(
                weighted_prediction
            )

        rows.append(
            row
        )

    return pd.DataFrame(
        rows
    )


# ============================================================
# METRICS
# ============================================================

def r_squared(
    y,
    yhat
):

    y = np.asarray(
        y,
        dtype=float
    )

    yhat = np.asarray(
        yhat,
        dtype=float
    )

    ss_res = np.sum(
        (
            y - yhat
        ) ** 2
    )

    ss_tot = np.sum(
        (
            y
            -
            np.mean(y)
        ) ** 2
    )

    if ss_tot == 0:
        return np.nan

    return float(
        1.0
        -
        ss_res / ss_tot
    )


def standard_metrics(
    y,
    yhat
):

    residual = (
        y - yhat
    )

    return {
        "R2":
            r_squared(
                y,
                yhat
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
# VALIDATION BINOMIAL LOG LOSS
# ============================================================

def validation_logloss(
    validation_df,
    prediction_column
):

    p = np.clip(
        validation_df[
            prediction_column
        ].to_numpy(
            dtype=float
        ),
        1e-12,
        1.0 - 1e-12
    )

    successes = validation_df[
        "successes"
    ].to_numpy(
        dtype=float
    )

    trials = validation_df[
        "trials"
    ].to_numpy(
        dtype=float
    )

    failures = (
        trials
        -
        successes
    )

    nll = -np.sum(
        successes
        *
        np.log(p)
        +
        failures
        *
        np.log(
            1.0 - p
        )
    )

    return float(
        nll
        /
        np.sum(
            trials
        )
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print(
        "=========================================="
    )

    print(
        "EXPERIMENT 6 — STRICT VALIDATION"
    )

    print(
        "=========================================="
    )

    # ========================================================
    # 1. TRAIN DATA
    # ========================================================

    train_df = generate_dataset(
        split="train",
        graph_types=TRAIN_GRAPH_TYPES,
        n_values=TRAIN_N,
        noise_values=TRAIN_NOISE,
        amplitude_values=TRAIN_AMPLITUDE,
        repetitions=TRAIN_REPETITIONS,
        seed=MASTER_SEED
    )

    train_file = os.path.join(
        OUTPUT_DIR,
        "train_grouped.csv"
    )

    train_df.to_csv(
        train_file,
        index=False
    )

    # ========================================================
    # 2. VALIDATION DATA
    # ========================================================

    valid_df = generate_dataset(
        split="validation",
        graph_types=VALID_GRAPH_TYPES,
        n_values=VALID_N,
        noise_values=VALID_NOISE,
        amplitude_values=VALID_AMPLITUDE,
        repetitions=VALID_REPETITIONS,
        seed=MASTER_SEED + 100000
    )

    valid_file = os.path.join(
        OUTPUT_DIR,
        "validation_grouped.csv"
    )

    valid_df.to_csv(
        valid_file,
        index=False
    )

    # ========================================================
    # 3. FIT THREE MODELS
    # ========================================================

    criteria = [
        "R_mean",
        "R_local",
        "R_full",
    ]

    fits = []

    for criterion in criteria:

        fit = fit_model(
            train_df,
            criterion
        )

        fits.append(
            fit
        )

    fits_df = pd.DataFrame(
        fits
    )

    fits_file = os.path.join(
        OUTPUT_DIR,
        "fitted_parameters.csv"
    )

    fits_df.to_csv(
        fits_file,
        index=False
    )

    print()
    print(
        "=========================================="
    )

    print(
        "FITTED PARAMETERS — TRAIN ONLY"
    )

    print(
        "=========================================="
    )

    print(
        fits_df.to_string(
            index=False
        )
    )

    # ========================================================
    # 4. APPLY TO VALIDATION
    # ========================================================

    prediction_columns = []

    for _, fit in fits_df.iterrows():

        criterion = fit[
            "criterion"
        ]

        prediction_column = (
            "pred_"
            +
            criterion
        )

        prediction_columns.append(
            prediction_column
        )

        apply_model(
            valid_df,
            feature_column=criterion,
            a=float(
                fit["a"]
            ),
            b=float(
                fit["b"]
            ),
            prediction_column=prediction_column
        )

    valid_predictions_file = os.path.join(
        OUTPUT_DIR,
        "validation_group_predictions.csv"
    )

    valid_df.to_csv(
        valid_predictions_file,
        index=False
    )

    # ========================================================
    # 5. AGGREGATE CONDITION-LEVEL P_D
    # ========================================================

    condition_df = aggregate_predictions(
        valid_df,
        prediction_columns
    )

    condition_file = os.path.join(
        OUTPUT_DIR,
        "validation_condition_predictions.csv"
    )

    condition_df.to_csv(
        condition_file,
        index=False
    )

    # ========================================================
    # 6. GLOBAL VALIDATION METRICS
    # ========================================================

    validation_rows = []

    observed = condition_df[
        "observed_pd"
    ].to_numpy()

    for criterion, prediction_column in zip(
        criteria,
        prediction_columns
    ):

        predicted = condition_df[
            prediction_column
        ].to_numpy()

        metrics = standard_metrics(
            observed,
            predicted
        )

        group_logloss = (
            validation_logloss(
                valid_df,
                prediction_column
            )
        )

        validation_rows.append(
            {
                "criterion":
                    criterion,

                "R2_validation":
                    metrics["R2"],

                "MAE_validation":
                    metrics["MAE"],

                "RMSE_validation":
                    metrics["RMSE"],

                "bias_validation":
                    metrics["bias"],

                "binomial_logloss_per_trial":
                    group_logloss
            }
        )

    validation_metrics = pd.DataFrame(
        validation_rows
    )

    metrics_file = os.path.join(
        OUTPUT_DIR,
        "validation_metrics.csv"
    )

    validation_metrics.to_csv(
        metrics_file,
        index=False
    )

    print()
    print(
        "=========================================="
    )

    print(
        "GLOBAL VALIDATION RESULTS"
    )

    print(
        "=========================================="
    )

    print(
        validation_metrics.to_string(
            index=False
        )
    )

    # ========================================================
    # 7. VALIDATION BY TOPOLOGY
    # ========================================================

    topology_rows = []

    for graph_type in VALID_GRAPH_TYPES:

        part = condition_df[
            condition_df[
                "graph_type"
            ]
            ==
            graph_type
        ]

        obs = part[
            "observed_pd"
        ].to_numpy()

        for criterion, prediction_column in zip(
            criteria,
            prediction_columns
        ):

            pred = part[
                prediction_column
            ].to_numpy()

            metrics = standard_metrics(
                obs,
                pred
            )

            topology_rows.append(
                {
                    "graph_type":
                        graph_type,

                    "criterion":
                        criterion,

                    "R2":
                        metrics["R2"],

                    "MAE":
                        metrics["MAE"],

                    "RMSE":
                        metrics["RMSE"],

                    "bias":
                        metrics["bias"]
                }
            )

    topology_metrics = pd.DataFrame(
        topology_rows
    )

    topology_file = os.path.join(
        OUTPUT_DIR,
        "validation_metrics_by_topology.csv"
    )

    topology_metrics.to_csv(
        topology_file,
        index=False
    )

    print()
    print(
        "=========================================="
    )

    print(
        "VALIDATION BY TOPOLOGY"
    )

    print(
        "=========================================="
    )

    print(
        topology_metrics.to_string(
            index=False
        )
    )

    # ========================================================
    # 8. FALSE ALARM CONTROL
    # ========================================================

    pfa_summary = (
        condition_df.groupby(
            "graph_type"
        )[
            "mean_pfa"
        ]
        .agg(
            [
                "mean",
                "min",
                "max",
                "std"
            ]
        )
        .reset_index()
    )

    pfa_file = os.path.join(
        OUTPUT_DIR,
        "validation_pfa.csv"
    )

    pfa_summary.to_csv(
        pfa_file,
        index=False
    )

    print()
    print(
        "=========================================="
    )

    print(
        "P_FA VALIDATION"
    )

    print(
        "=========================================="
    )

    print(
        pfa_summary.to_string(
            index=False
        )
    )

    # ========================================================
    # 9. PLOTS
    # ========================================================

    for criterion, prediction_column in zip(
        criteria,
        prediction_columns
    ):

        metrics_row = validation_metrics[
            validation_metrics[
                "criterion"
            ]
            ==
            criterion
        ].iloc[0]

        plt.figure(
            figsize=(7, 7)
        )

        for graph_type in VALID_GRAPH_TYPES:

            part = condition_df[
                condition_df[
                    "graph_type"
                ]
                ==
                graph_type
            ]

            plt.scatter(
                part[
                    prediction_column
                ],
                part[
                    "observed_pd"
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
            "Predicted P_D"
        )

        plt.ylabel(
            "Observed P_D"
        )

        plt.title(
            f"{criterion}: validation, "
            f"R²="
            f"{metrics_row['R2_validation']:.3f}"
        )

        plt.legend()

        plt.grid(
            alpha=0.25
        )

        plt.tight_layout()

        plt.savefig(
            os.path.join(
                OUTPUT_DIR,
                f"{criterion}_observed_vs_predicted.png"
            ),
            dpi=300
        )

        plt.close()

    # ========================================================
    # 10. ERROR vs DEGREE HETEROGENEITY
    # ========================================================

    for criterion, prediction_column in zip(
        criteria,
        prediction_columns
    ):

        error = np.abs(
            condition_df[
                "observed_pd"
            ]
            -
            condition_df[
                prediction_column
            ]
        )

        plt.figure(
            figsize=(8, 6)
        )

        plt.scatter(
            condition_df[
                "degree_cv"
            ],
            error,
            alpha=0.65
        )

        plt.xlabel(
            "Coefficient of variation "
            "of node degree"
        )

        plt.ylabel(
            "Absolute prediction error"
        )

        plt.title(
            f"{criterion}: error vs "
            "degree heterogeneity"
        )

        plt.grid(
            alpha=0.25
        )

        plt.tight_layout()

        plt.savefig(
            os.path.join(
                OUTPUT_DIR,
                f"{criterion}_error_vs_degree_cv.png"
            ),
            dpi=300
        )

        plt.close()

    # ========================================================
    # FINISH
    # ========================================================

    print()
    print(
        "=========================================="
    )

    print(
        "EXPERIMENT 6 COMPLETED"
    )

    print(
        "=========================================="
    )

    print()

    print(
        f"Results saved to: {OUTPUT_DIR}/"
    )


if __name__ == "__main__":
    main()
