import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from config import ExperimentConfig

from graph_model import (
    build_graph,
    laplacian_matrix,
    graph_energy_batch,
)

from detector import (
    empirical_threshold,
    gaussian_threshold,
    false_alarm_rate,
    detection_probability,
)


# ============================================================
# DATA GENERATION
# ============================================================

def generate_background(
    rng,
    num_samples,
    num_sensors,
    noise_sigma
):
    """
    H0:
        x = epsilon,
        epsilon ~ N(0, sigma^2 I)
    """

    return rng.normal(
        loc=0.0,
        scale=noise_sigma,
        size=(num_samples, num_sensors)
    )


def generate_anomalies(
    rng,
    num_samples,
    num_sensors,
    noise_sigma,
    anomaly_amplitude,
    anomaly_width=1
):
    """
    H1:
        x = s + epsilon.

    s — локализованная аномалия.
    """

    X = rng.normal(
        loc=0.0,
        scale=noise_sigma,
        size=(num_samples, num_sensors)
    )

    for sample_idx in range(num_samples):

        center = rng.integers(
            0,
            num_sensors
        )

        for offset in range(anomaly_width):

            node = (
                center + offset
            ) % num_sensors

            X[
                sample_idx,
                node
            ] += anomaly_amplitude

    return X


# ============================================================
# CONFIDENCE INTERVAL
# ============================================================

def confidence_interval_95(values):
    """
    Приближённый 95% ДИ для среднего.
    """

    values = np.asarray(values)

    mean = np.mean(values)

    if len(values) <= 1:
        return mean, 0.0

    std = np.std(
        values,
        ddof=1
    )

    half_width = (
        1.96
        * std
        / np.sqrt(len(values))
    )

    return mean, half_width


# ============================================================
# SINGLE REPETITION
# ============================================================

def run_single_repetition(
    cfg,
    L,
    noise,
    fixed_tau,
    rng
):
    # -----------------------------
    # TRAIN / CALIBRATION BACKGROUND
    # -----------------------------

    X_cal = generate_background(
        rng=rng,
        num_samples=cfg.background_samples,
        num_sensors=cfg.num_sensors,
        noise_sigma=noise,
    )

    E_cal = graph_energy_batch(
        X_cal,
        L
    )

    tau_empirical = empirical_threshold(
        E_cal,
        cfg.false_alarm_alpha
    )

    tau_gaussian = gaussian_threshold(
        E_cal,
        cfg.false_alarm_alpha
    )

    # -----------------------------
    # INDEPENDENT H0 TEST
    # -----------------------------

    X_normal = generate_background(
        rng=rng,
        num_samples=cfg.test_normal_samples,
        num_sensors=cfg.num_sensors,
        noise_sigma=noise,
    )

    E_normal = graph_energy_batch(
        X_normal,
        L
    )

    # -----------------------------
    # INDEPENDENT H1 TEST
    # -----------------------------

    X_anomaly = generate_anomalies(
        rng=rng,
        num_samples=cfg.test_anomaly_samples,
        num_sensors=cfg.num_sensors,
        noise_sigma=noise,
        anomaly_amplitude=cfg.anomaly_amplitude,
        anomaly_width=cfg.anomaly_width,
    )

    E_anomaly = graph_energy_batch(
        X_anomaly,
        L
    )

    # -----------------------------
    # METRICS
    # -----------------------------

    result = {
        "noise": noise,

        "fixed_tau": fixed_tau,
        "empirical_tau": tau_empirical,
        "gaussian_tau": tau_gaussian,

        "fixed_pfa":
            false_alarm_rate(
                E_normal,
                fixed_tau
            ),

        "empirical_pfa":
            false_alarm_rate(
                E_normal,
                tau_empirical
            ),

        "gaussian_pfa":
            false_alarm_rate(
                E_normal,
                tau_gaussian
            ),

        "fixed_pd":
            detection_probability(
                E_anomaly,
                fixed_tau
            ),

        "empirical_pd":
            detection_probability(
                E_anomaly,
                tau_empirical
            ),

        "gaussian_pd":
            detection_probability(
                E_anomaly,
                tau_gaussian
            ),
    }

    return result


# ============================================================
# MAIN
# ============================================================

def main():

    cfg = ExperimentConfig()

    os.makedirs(
        cfg.results_dir,
        exist_ok=True
    )

    # -----------------------------
    # GRAPH
    # -----------------------------

    G = build_graph(
        num_nodes=cfg.num_sensors,
        graph_type=cfg.graph_type
    )

    L = laplacian_matrix(G)

    print(
        f"Граф: {cfg.graph_type}"
    )

    print(
        f"Число сенсоров: "
        f"{cfg.num_sensors}"
    )

    print(
        f"Число рёбер: "
        f"{G.number_of_edges()}"
    )

    # ========================================================
    # FIXED THRESHOLD CALIBRATION
    # ========================================================

    baseline_rng = np.random.default_rng(
        cfg.random_seed
    )

    X_baseline = generate_background(
        rng=baseline_rng,
        num_samples=cfg.background_samples * 3,
        num_sensors=cfg.num_sensors,
        noise_sigma=cfg.baseline_noise,
    )

    E_baseline = graph_energy_batch(
        X_baseline,
        L
    )

    fixed_tau = empirical_threshold(
        E_baseline,
        cfg.false_alarm_alpha
    )

    print(
        f"\nFixed threshold calibrated "
        f"at sigma={cfg.baseline_noise}:"
    )

    print(
        f"tau_fixed = {fixed_tau:.6f}"
    )

    # ========================================================
    # MONTE CARLO
    # ========================================================

    all_results = []

    master_rng = np.random.default_rng(
        cfg.random_seed + 1000
    )

    for noise in cfg.noise_levels:

        print(
            f"\nNoise sigma = {noise}"
        )

        for repetition in range(
            cfg.repetitions
        ):

            seed = master_rng.integers(
                0,
                2**32 - 1
            )

            rng = np.random.default_rng(
                seed
            )

            result = run_single_repetition(
                cfg=cfg,
                L=L,
                noise=noise,
                fixed_tau=fixed_tau,
                rng=rng,
            )

            result["repetition"] = repetition

            all_results.append(
                result
            )

    raw_df = pd.DataFrame(
        all_results
    )

    raw_path = os.path.join(
        cfg.results_dir,
        "raw_results.csv"
    )

    raw_df.to_csv(
        raw_path,
        index=False
    )

    # ========================================================
    # AGGREGATION
    # ========================================================

    metrics = [
        "fixed_pfa",
        "empirical_pfa",
        "gaussian_pfa",
        "fixed_pd",
        "empirical_pd",
        "gaussian_pd",
        "fixed_tau",
        "empirical_tau",
        "gaussian_tau",
    ]

    summary_rows = []

    for noise in cfg.noise_levels:

        subset = raw_df[
            raw_df["noise"] == noise
        ]

        row = {
            "noise": noise
        }

        for metric in metrics:

            mean, ci = confidence_interval_95(
                subset[metric].values
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

    summary_df = pd.DataFrame(
        summary_rows
    )

    summary_path = os.path.join(
        cfg.results_dir,
        "summary_results.csv"
    )

    summary_df.to_csv(
        summary_path,
        index=False
    )

    print("\nSUMMARY")
    print(summary_df.to_string(index=False))

    # ========================================================
    # PLOT 1: FALSE ALARM
    # ========================================================

    plt.figure(figsize=(8, 5))

    plt.errorbar(
        summary_df["noise"],
        summary_df["fixed_pfa_mean"],
        yerr=summary_df["fixed_pfa_ci95"],
        marker="o",
        label="Fixed"
    )

    plt.errorbar(
        summary_df["noise"],
        summary_df["empirical_pfa_mean"],
        yerr=summary_df["empirical_pfa_ci95"],
        marker="o",
        label="Adaptive empirical"
    )

    plt.errorbar(
        summary_df["noise"],
        summary_df["gaussian_pfa_mean"],
        yerr=summary_df["gaussian_pfa_ci95"],
        marker="o",
        label="Adaptive Gaussian"
    )

    plt.axhline(
        cfg.false_alarm_alpha,
        linestyle="--",
        label="Target alpha"
    )

    plt.xlabel("Noise standard deviation")
    plt.ylabel("False alarm probability")

    plt.title(
        "False Alarm Probability vs Noise"
    )

    plt.legend()
    plt.tight_layout()

    plt.savefig(
        os.path.join(
            cfg.results_dir,
            "pfa_vs_noise.png"
        ),
        dpi=300
    )

    plt.close()

    # ========================================================
    # PLOT 2: DETECTION PROBABILITY
    # ========================================================

    plt.figure(figsize=(8, 5))

    plt.errorbar(
        summary_df["noise"],
        summary_df["fixed_pd_mean"],
        yerr=summary_df["fixed_pd_ci95"],
        marker="o",
        label="Fixed"
    )

    plt.errorbar(
        summary_df["noise"],
        summary_df["empirical_pd_mean"],
        yerr=summary_df["empirical_pd_ci95"],
        marker="o",
        label="Adaptive empirical"
    )

    plt.errorbar(
        summary_df["noise"],
        summary_df["gaussian_pd_mean"],
        yerr=summary_df["gaussian_pd_ci95"],
        marker="o",
        label="Adaptive Gaussian"
    )

    plt.xlabel("Noise standard deviation")
    plt.ylabel("Detection probability")

    plt.title(
        "Detection Probability vs Noise"
    )

    plt.legend()
    plt.tight_layout()

    plt.savefig(
        os.path.join(
            cfg.results_dir,
            "pd_vs_noise.png"
        ),
        dpi=300
    )

    plt.close()

    # ========================================================
    # PLOT 3: THRESHOLD
    # ========================================================

    plt.figure(figsize=(8, 5))

    plt.plot(
        summary_df["noise"],
        summary_df["fixed_tau_mean"],
        marker="o",
        label="Fixed"
    )

    plt.plot(
        summary_df["noise"],
        summary_df["empirical_tau_mean"],
        marker="o",
        label="Adaptive empirical"
    )

    plt.plot(
        summary_df["noise"],
        summary_df["gaussian_tau_mean"],
        marker="o",
        label="Adaptive Gaussian"
    )

    plt.xlabel("Noise standard deviation")
    plt.ylabel("Threshold tau")

    plt.title(
        "Threshold Adaptation vs Noise"
    )

    plt.legend()
    plt.tight_layout()

    plt.savefig(
        os.path.join(
            cfg.results_dir,
            "tau_vs_noise.png"
        ),
        dpi=300
    )

    plt.close()

    print(
        "\nЭксперимент завершён."
    )

    print(
        f"Результаты сохранены в "
        f"'{cfg.results_dir}/'"
    )


if __name__ == "__main__":
    main()