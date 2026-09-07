import os
import math
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
# EXPERIMENT 7
# SPECTRAL PREDICTION OF DETECTION PROBABILITY
# ============================================================


OUTPUT_DIR = "results/experiment_07"

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
    "random_regular",
]

NUM_SENSOR_VALUES = [
    15,
    30,
    60,
]

NOISE_LEVELS = [
    0.15,
    0.25,
    0.35,
]

ANOMALY_AMPLITUDES = [
    1.0,
    1.5,
    2.0,
    3.0,
]

REPETITIONS = 5

BACKGROUND_SAMPLES = 10000
ANOMALY_SAMPLES = 10000

ALPHA = 0.01

MASTER_SEED = 20260915

EIGENVALUE_TOL = 1e-10


# ============================================================
# IMHOF NUMERICAL SETTINGS
# ============================================================

IMHOF_EPSABS = 1e-7
IMHOF_EPSREL = 1e-7
IMHOF_LIMIT = 1000

PROBABILITY_CLIP = 1e-10


# ============================================================
# RANDOM GENERATOR
# ============================================================

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

        for _ in range(100):

            seed = int(
                rng.integers(
                    0,
                    2**31 - 1
                )
            )

            radius = 1.5 * np.sqrt(
                np.log(n) / n
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
                "Failed to generate connected "
                "random_geometric graph"
            )

    elif graph_type == "random_regular":

        degree = min(
            4,
            n - 1
        )

        if (
            n * degree
        ) % 2 != 0:

            degree -= 1

        for _ in range(100):

            seed = int(
                rng.integers(
                    0,
                    2**31 - 1
                )
            )

            G = nx.random_regular_graph(
                d=degree,
                n=n,
                seed=seed
            )

            if nx.is_connected(G):

                break

        else:

            raise RuntimeError(
                "Failed to generate connected "
                "random_regular graph"
            )

    else:

        raise ValueError(
            f"Unknown graph type: {graph_type}"
        )

    return G


# ============================================================
# LAPLACIAN
# ============================================================

def graph_laplacian(
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
        L,
        degrees
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
# SPECTRAL DECOMPOSITION
# ============================================================

def spectral_decomposition(
    L
):

    eigenvalues, eigenvectors = np.linalg.eigh(
        L
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
        positive_eigenvalues,
        positive_eigenvectors
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

    mask = (
        np.abs(weights)
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
            q < 0
        )

    if q <= 0:

        return 1.0

    # --------------------------------------------------------
    # Imhof integrand
    #
    # theta(u)
    #
    # rho(u)
    # --------------------------------------------------------

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
                (
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

        return (
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

    # --------------------------------------------------------
    # Integral over [0, infinity)
    # --------------------------------------------------------

    with warnings.catch_warnings():

        warnings.simplefilter(
            "ignore"
        )

        integral, error = quad(
            integrand,
            0.0,
            np.inf,
            epsabs=IMHOF_EPSABS,
            epsrel=IMHOF_EPSREL,
            limit=IMHOF_LIMIT
        )

    sf = (
        0.5
        +
        integral
        /
        np.pi
    )

    sf = float(
        np.clip(
            sf,
            0.0,
            1.0
        )
    )

    return sf


# ============================================================
# SPECTRAL H0 SURVIVAL FUNCTION
# ============================================================

def spectral_h0_sf(
    q,
    eigenvalues,
    sigma
):

    weights = (
        sigma ** 2
        *
        eigenvalues
    )

    noncentralities = np.zeros_like(
        weights
    )

    return imhof_sf(
        q=q,
        weights=weights,
        noncentralities=noncentralities
    )


# ============================================================
# SPECTRAL H1 SURVIVAL FUNCTION
# ============================================================

def spectral_h1_sf(
    q,
    eigenvalues,
    eigenvectors,
    sigma,
    amplitude,
    anomaly_node
):

    weights = (
        sigma ** 2
        *
        eigenvalues
    )

    node_components = (
        eigenvectors[
            anomaly_node,
            :
        ]
    )

    noncentralities = (
        (
            amplitude
            /
            sigma
        ) ** 2
        *
        node_components ** 2
    )

    return imhof_sf(
        q=q,
        weights=weights,
        noncentralities=noncentralities
    )


# ============================================================
# SPECTRAL THRESHOLD
# ============================================================

def spectral_threshold(
    eigenvalues,
    sigma,
    alpha
):

    mean_q = (
        sigma ** 2
        *
        np.sum(
            eigenvalues
        )
    )

    variance_q = (
        2.0
        *
        sigma ** 4
        *
        np.sum(
            eigenvalues ** 2
        )
    )

    std_q = np.sqrt(
        variance_q
    )

    lower = 0.0

    upper = (
        mean_q
        +
        10.0
        *
        std_q
    )

    # Expand upper bound if needed

    for _ in range(20):

        sf_upper = spectral_h0_sf(
            upper,
            eigenvalues,
            sigma
        )

        if sf_upper < alpha:

            break

        upper *= 2.0

    else:

        raise RuntimeError(
            "Could not bracket spectral threshold"
        )

    def root_function(
        q
    ):

        return (
            spectral_h0_sf(
                q,
                eigenvalues,
                sigma
            )
            -
            alpha
        )

    tau = brentq(
        root_function,
        lower,
        upper,
        xtol=1e-8,
        rtol=1e-8,
        maxiter=100
    )

    return float(
        tau
    )


# ============================================================
# MODEL METRICS
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

    bias = np.mean(
        observed
        -
        predicted
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
            float(
                bias
            ),
    }


# ============================================================
# ONE GRAPH INSTANCE
# ============================================================

def run_graph_instance(
    graph_type,
    n,
    sigma,
    amplitude,
    repetition,
    rng
):

    G = generate_graph(
        graph_type,
        n,
        rng
    )

    L, degrees = graph_laplacian(
        G
    )

    eigenvalues, eigenvectors = spectral_decomposition(
        L
    )

    # --------------------------------------------------------
    # Background Monte Carlo
    # --------------------------------------------------------

    background = rng.normal(
        loc=0.0,
        scale=sigma,
        size=(
            BACKGROUND_SAMPLES,
            n
        )
    )

    q_background = graph_energy_samples(
        background,
        L
    )

    tau_empirical = float(
        np.quantile(
            q_background,
            1.0 - ALPHA
        )
    )

    pfa_empirical = float(
        np.mean(
            q_background
            >
            tau_empirical
        )
    )

    # --------------------------------------------------------
    # Spectral threshold
    # --------------------------------------------------------

    tau_spectral = spectral_threshold(
        eigenvalues,
        sigma,
        ALPHA
    )

    pfa_spectral_at_empirical_tau = (
        spectral_h0_sf(
            tau_empirical,
            eigenvalues,
            sigma
        )
    )

    pfa_spectral_at_spectral_tau = (
        spectral_h0_sf(
            tau_spectral,
            eigenvalues,
            sigma
        )
    )

    # --------------------------------------------------------
    # Monte Carlo anomalies
    # --------------------------------------------------------

    anomaly_nodes = rng.integers(
        low=0,
        high=n,
        size=ANOMALY_SAMPLES
    )

    anomaly_noise = rng.normal(
        loc=0.0,
        scale=sigma,
        size=(
            ANOMALY_SAMPLES,
            n
        )
    )

    anomaly_X = anomaly_noise.copy()

    anomaly_X[
        np.arange(
            ANOMALY_SAMPLES
        ),
        anomaly_nodes
    ] += amplitude

    q_anomaly = graph_energy_samples(
        anomaly_X,
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
    # Spectral prediction averaged over nodes
    # --------------------------------------------------------

    node_pd_emp_tau = []

    node_pd_spec_tau = []

    for k in range(n):

        pd_emp_k = spectral_h1_sf(
            q=tau_empirical,
            eigenvalues=eigenvalues,
            eigenvectors=eigenvectors,
            sigma=sigma,
            amplitude=amplitude,
            anomaly_node=k
        )

        pd_spec_k = spectral_h1_sf(
            q=tau_spectral,
            eigenvalues=eigenvalues,
            eigenvectors=eigenvectors,
            sigma=sigma,
            amplitude=amplitude,
            anomaly_node=k
        )

        node_pd_emp_tau.append(
            pd_emp_k
        )

        node_pd_spec_tau.append(
            pd_spec_k
        )

    pd_spectral_emp_tau = float(
        np.mean(
            node_pd_emp_tau
        )
    )

    pd_spectral_spec_tau = float(
        np.mean(
            node_pd_spec_tau
        )
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
            L @ L
        )
    )

    mean_degree = float(
        np.mean(
            degrees
        )
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
        if mean_degree > 0
        else 0.0
    )

    lambda2 = (
        float(
            eigenvalues[0]
        )
        if len(eigenvalues) > 0
        else 0.0
    )

    lambda_max = (
        float(
            eigenvalues[-1]
        )
        if len(eigenvalues) > 0
        else 0.0
    )

    return {

        "graph_type":
            graph_type,

        "n":
            n,

        "sigma":
            sigma,

        "amplitude":
            amplitude,

        "repetition":
            repetition,

        "edges":
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

        "lambda2":
            lambda2,

        "lambda_max":
            lambda_max,

        "tau_empirical":
            tau_empirical,

        "tau_spectral":
            tau_spectral,

        "tau_relative_error":
            (
                tau_spectral
                -
                tau_empirical
            )
            /
            tau_empirical,

        "pfa_empirical":
            pfa_empirical,

        "pfa_spectral_at_emp_tau":
            pfa_spectral_at_empirical_tau,

        "pfa_spectral_at_spec_tau":
            pfa_spectral_at_spectral_tau,

        "pd_mc_emp_tau":
            pd_mc_emp_tau,

        "pd_spectral_emp_tau":
            pd_spectral_emp_tau,

        "pd_error_emp_tau":
            (
                pd_mc_emp_tau
                -
                pd_spectral_emp_tau
            ),

        "pd_mc_spec_tau":
            pd_mc_spec_tau,

        "pd_spectral_spec_tau":
            pd_spectral_spec_tau,

        "pd_error_spec_tau":
            (
                pd_mc_spec_tau
                -
                pd_spectral_spec_tau
            ),
    }


# ============================================================
# MAIN EXPERIMENT
# ============================================================

def main():

    print()
    print(
        "============================================"
    )

    print(
        "EXPERIMENT 7"
    )

    print(
        "SPECTRAL DETECTION PROBABILITY"
    )

    print(
        "============================================"
    )

    total_runs = (
        len(
            GRAPH_TYPES
        )
        *
        len(
            NUM_SENSOR_VALUES
        )
        *
        len(
            NOISE_LEVELS
        )
        *
        len(
            ANOMALY_AMPLITUDES
        )
        *
        REPETITIONS
    )

    print(
        f"Total graph instances: {total_runs}"
    )

    rows = []

    run_counter = 0

    for graph_type in GRAPH_TYPES:

        for n in NUM_SENSOR_VALUES:

            for sigma in NOISE_LEVELS:

                for amplitude in ANOMALY_AMPLITUDES:

                    for repetition in range(
                        REPETITIONS
                    ):

                        run_counter += 1

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
                            f"[{run_counter}/{total_runs}] "
                            f"{graph_type}, "
                            f"n={n}, "
                            f"sigma={sigma}, "
                            f"A={amplitude}, "
                            f"rep={repetition}"
                        )

                        result = run_graph_instance(
                            graph_type=graph_type,
                            n=n,
                            sigma=sigma,
                            amplitude=amplitude,
                            repetition=repetition,
                            rng=rng
                        )

                        rows.append(
                            result
                        )

    # ========================================================
    # RAW DATA
    # ========================================================

    raw_df = pd.DataFrame(
        rows
    )

    raw_file = os.path.join(
        OUTPUT_DIR,
        "experiment7_raw.csv"
    )

    raw_df.to_csv(
        raw_file,
        index=False
    )

    # ========================================================
    # AGGREGATED CONDITIONS
    # ========================================================

    group_columns = [
        "graph_type",
        "n",
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

            pfa_empirical_mean=(
                "pfa_empirical",
                "mean"
            ),

            pfa_spectral_emp_mean=(
                "pfa_spectral_at_emp_tau",
                "mean"
            ),

            pfa_spectral_spec_mean=(
                "pfa_spectral_at_spec_tau",
                "mean"
            ),

            pd_mc_emp_mean=(
                "pd_mc_emp_tau",
                "mean"
            ),

            pd_spectral_emp_mean=(
                "pd_spectral_emp_tau",
                "mean"
            ),

            pd_mc_spec_mean=(
                "pd_mc_spec_tau",
                "mean"
            ),

            pd_spectral_spec_mean=(
                "pd_spectral_spec_tau",
                "mean"
            ),

            degree_cv_mean=(
                "degree_cv",
                "mean"
            ),
        )
    )

    summary_file = os.path.join(
        OUTPUT_DIR,
        "experiment7_summary.csv"
    )

    summary_df.to_csv(
        summary_file,
        index=False
    )

    # ========================================================
    # GLOBAL METRICS
    # ========================================================

    metrics_emp = regression_metrics(
        summary_df[
            "pd_mc_emp_mean"
        ],
        summary_df[
            "pd_spectral_emp_mean"
        ]
    )

    metrics_spec = regression_metrics(
        summary_df[
            "pd_mc_spec_mean"
        ],
        summary_df[
            "pd_spectral_spec_mean"
        ]
    )

    metrics_rows = [
        {
            "method":
                "spectral_H1_empirical_tau",

            **metrics_emp
        },

        {
            "method":
                "fully_spectral",

            **metrics_spec
        }
    ]

    metrics_df = pd.DataFrame(
        metrics_rows
    )

    metrics_file = os.path.join(
        OUTPUT_DIR,
        "experiment7_metrics.csv"
    )

    metrics_df.to_csv(
        metrics_file,
        index=False
    )

    # ========================================================
    # METRICS BY TOPOLOGY
    # ========================================================

    topology_rows = []

    for graph_type in GRAPH_TYPES:

        sub = summary_df[
            summary_df[
                "graph_type"
            ]
            ==
            graph_type
        ]

        metrics_emp_topology = (
            regression_metrics(
                sub[
                    "pd_mc_emp_mean"
                ],
                sub[
                    "pd_spectral_emp_mean"
                ]
            )
        )

        metrics_spec_topology = (
            regression_metrics(
                sub[
                    "pd_mc_spec_mean"
                ],
                sub[
                    "pd_spectral_spec_mean"
                ]
            )
        )

        topology_rows.append(
            {
                "graph_type":
                    graph_type,

                "method":
                    "spectral_H1_empirical_tau",

                **metrics_emp_topology
            }
        )

        topology_rows.append(
            {
                "graph_type":
                    graph_type,

                "method":
                    "fully_spectral",

                **metrics_spec_topology
            }
        )

    topology_df = pd.DataFrame(
        topology_rows
    )

    topology_file = os.path.join(
        OUTPUT_DIR,
        "experiment7_metrics_by_topology.csv"
    )

    topology_df.to_csv(
        topology_file,
        index=False
    )

    # ========================================================
    # PLOTS
    # ========================================================

    plt.figure(
        figsize=(7, 7)
    )

    plt.scatter(
        summary_df[
            "pd_mc_emp_mean"
        ],
        summary_df[
            "pd_spectral_emp_mean"
        ],
        alpha=0.7
    )

    plt.plot(
        [0, 1],
        [0, 1],
        linestyle="--"
    )

    plt.xlabel(
        "Observed Monte Carlo P_D"
    )

    plt.ylabel(
        "Spectral predicted P_D"
    )

    plt.title(
        "Experiment 7: empirical threshold"
    )

    plt.grid(
        alpha=0.25
    )

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            OUTPUT_DIR,
            "spectral_empirical_tau.png"
        ),
        dpi=300
    )

    plt.close()

    plt.figure(
        figsize=(7, 7)
    )

    plt.scatter(
        summary_df[
            "pd_mc_spec_mean"
        ],
        summary_df[
            "pd_spectral_spec_mean"
        ],
        alpha=0.7
    )

    plt.plot(
        [0, 1],
        [0, 1],
        linestyle="--"
    )

    plt.xlabel(
        "Observed Monte Carlo P_D"
    )

    plt.ylabel(
        "Fully spectral predicted P_D"
    )

    plt.title(
        "Experiment 7: fully spectral"
    )

    plt.grid(
        alpha=0.25
    )

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            OUTPUT_DIR,
            "fully_spectral.png"
        ),
        dpi=300
    )

    plt.close()

    # ========================================================
    # PRINT RESULTS
    # ========================================================

    print()
    print(
        "============================================"
    )

    print(
        "GLOBAL METRICS"
    )

    print(
        "============================================"
    )

    print(
        metrics_df.to_string(
            index=False
        )
    )

    print()
    print(
        "============================================"
    )

    print(
        "METRICS BY TOPOLOGY"
    )

    print(
        "============================================"
    )

    print(
        topology_df.to_string(
            index=False
        )
    )

    print()
    print(
        "============================================"
    )

    print(
        "THRESHOLD CHECK"
    )

    print(
        "============================================"
    )

    print(
        "Mean empirical PFA:",
        raw_df[
            "pfa_empirical"
        ].mean()
    )

    print(
        "Mean spectral PFA at empirical tau:",
        raw_df[
            "pfa_spectral_at_emp_tau"
        ].mean()
    )

    print(
        "Mean spectral PFA at spectral tau:",
        raw_df[
            "pfa_spectral_at_spec_tau"
        ].mean()
    )

    print(
        "Mean relative tau error:",
        raw_df[
            "tau_relative_error"
        ].mean()
    )

    print()
    print(
        "============================================"
    )

    print(
        "EXPERIMENT 7 COMPLETED"
    )

    print(
        "============================================"
    )

    print(
        f"Results saved to: {OUTPUT_DIR}/"
    )


if __name__ == "__main__":
    main()
