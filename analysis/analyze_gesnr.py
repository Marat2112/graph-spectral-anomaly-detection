import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.optimize import curve_fit
from scipy.stats import pearsonr, spearmanr


# ============================================================
# SETTINGS
# ============================================================

INPUT_FILE = os.path.join(
    "results",
    "experiment_02",
    "topology_summary.csv"
)

OUTPUT_DIR = "results/experiment_03"

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# ============================================================
# LOGISTIC MODEL
# ============================================================

def logistic_log_r(r, a, b):
    """
    Логистическая модель зависимости вероятности обнаружения
    от безразмерного критерия GESNR:

        P_D(R) =
        1 / (1 + exp(-(a * ln(R) + b)))

    Используем ln(R), поскольку значения R могут
    различаться на несколько порядков.
    """

    r = np.maximum(
        r,
        1e-12
    )

    return 1.0 / (
        1.0
        +
        np.exp(
            -(
                a * np.log(r)
                +
                b
            )
        )
    )


# ============================================================
# R^2
# ============================================================

def r_squared(y_true, y_pred):
    """
    Коэффициент детерминации R^2.
    """

    ss_res = np.sum(
        (y_true - y_pred) ** 2
    )

    ss_tot = np.sum(
        (
            y_true
            -
            np.mean(y_true)
        ) ** 2
    )

    if ss_tot == 0:
        return np.nan

    return (
        1.0
        -
        ss_res / ss_tot
    )


# ============================================================
# CRITICAL GESNR
# ============================================================

def critical_r(probability, a, b):
    """
    Вычисляет такое значение GESNR,
    при котором логистическая модель даёт
    заданную вероятность обнаружения.

    Из

        P_D =
        1 / (1 + exp(-(a ln(R) + b)))

    следует

        R_p =
        exp(
            [logit(p) - b] / a
        ).
    """

    if not 0.0 < probability < 1.0:
        raise ValueError(
            "Probability должна принадлежать интервалу (0, 1)."
        )

    logit = np.log(
        probability
        /
        (1.0 - probability)
    )

    return np.exp(
        (logit - b) / a
    )


# ============================================================
# MAIN
# ============================================================

def main():

    # --------------------------------------------------------
    # LOAD DATA
    # --------------------------------------------------------

    if not os.path.exists(INPUT_FILE):
        raise FileNotFoundError(
            f"Не найден файл: {INPUT_FILE}"
        )

    df = pd.read_csv(
        INPUT_FILE
    )

    print()
    print("===================================")
    print("GESNR ANALYSIS")
    print("===================================")

    print(
        f"Загружено строк: {len(df)}"
    )

    # --------------------------------------------------------
    # CHECK REQUIRED COLUMNS
    # --------------------------------------------------------

    required_columns = [
        "graph_type",
        "num_sensors",
        "noise",
        "amplitude",
        "mean_anomaly_degree_mean",
        "trace_L2_mean",
        "pd_mean"
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            "В topology_summary.csv отсутствуют столбцы: "
            + ", ".join(missing_columns)
        )

    # --------------------------------------------------------
    # CALCULATE GESNR
    # --------------------------------------------------------
    #
    #               A^2 * d
    # R = -------------------------------
    #       sigma^2 * sqrt(2 * tr(L^2))
    #
    # --------------------------------------------------------

    numerator = (
        df["amplitude"] ** 2
        *
        df["mean_anomaly_degree_mean"]
    )

    denominator = (
        df["noise"] ** 2
        *
        np.sqrt(
            2.0
            *
            df["trace_L2_mean"]
        )
    )

    df["gesnr"] = (
        numerator
        /
        denominator
    )

    # Логарифм GESNR нужен для
    # логистической модели.

    df["log_gesnr"] = np.log(
        np.maximum(
            df["gesnr"],
            1e-12
        )
    )

    # --------------------------------------------------------
    # BASIC INFORMATION
    # --------------------------------------------------------

    print()
    print("Диапазон GESNR:")

    print(
        f"min = {df['gesnr'].min():.6f}"
    )

    print(
        f"max = {df['gesnr'].max():.6f}"
    )

    print()
    print("Диапазон P_D:")

    print(
        f"min = {df['pd_mean'].min():.6f}"
    )

    print(
        f"max = {df['pd_mean'].max():.6f}"
    )

    # ========================================================
    # CORRELATION ANALYSIS
    # ========================================================

    pearson_r, pearson_p = pearsonr(
        df["gesnr"],
        df["pd_mean"]
    )

    spearman_r, spearman_p = spearmanr(
        df["gesnr"],
        df["pd_mean"]
    )

    print()
    print("===================================")
    print("КОРРЕЛЯЦИОННЫЙ АНАЛИЗ")
    print("===================================")

    print(
        f"Pearson r  = {pearson_r:.6f}"
    )

    print(
        f"Pearson p  = {pearson_p:.6e}"
    )

    print(
        f"Spearman rho = {spearman_r:.6f}"
    )

    print(
        f"Spearman p = {spearman_p:.6e}"
    )

    # ========================================================
    # LOGISTIC FIT
    # ========================================================

    x = df[
        "gesnr"
    ].to_numpy()

    y = df[
        "pd_mean"
    ].to_numpy()

    initial_guess = [
        2.0,
        -2.0
    ]

    params, covariance = curve_fit(
        logistic_log_r,
        x,
        y,
        p0=initial_guess,
        maxfev=20000
    )

    a_fit = params[0]
    b_fit = params[1]

    y_pred = logistic_log_r(
        x,
        a_fit,
        b_fit
    )

    r2 = r_squared(
        y,
        y_pred
    )

    # --------------------------------------------------------
    # CRITICAL GESNR VALUES
    # --------------------------------------------------------

    r50 = critical_r(
        0.50,
        a_fit,
        b_fit
    )

    r90 = critical_r(
        0.90,
        a_fit,
        b_fit
    )

    r95 = critical_r(
        0.95,
        a_fit,
        b_fit
    )

    # --------------------------------------------------------
    # PRINT MODEL RESULTS
    # --------------------------------------------------------

    print()
    print("===================================")
    print("ЛОГИСТИЧЕСКАЯ АППРОКСИМАЦИЯ")
    print("===================================")

    print(
        f"a = {a_fit:.6f}"
    )

    print(
        f"b = {b_fit:.6f}"
    )

    print(
        f"R^2 = {r2:.6f}"
    )

    print()

    print("Модель:")

    print(
        "P_D(R) = "
        "1 / "
        "(1 + exp(-(a * ln(R) + b)))"
    )

    print()
    print("===================================")
    print("КРИТИЧЕСКИЕ ЗНАЧЕНИЯ GESNR")
    print("===================================")

    print(
        f"GESNR_50 = {r50:.6f}"
    )

    print(
        f"GESNR_90 = {r90:.6f}"
    )

    print(
        f"GESNR_95 = {r95:.6f}"
    )

    # --------------------------------------------------------
    # ADD PREDICTIONS AND RESIDUALS
    # --------------------------------------------------------

    df["pd_predicted"] = (
        y_pred
    )

    df["residual"] = (
        df["pd_mean"]
        -
        df["pd_predicted"]
    )

    df["abs_residual"] = np.abs(
        df["residual"]
    )

    # ========================================================
    # ERROR METRICS
    # ========================================================

    mae = np.mean(
        np.abs(
            y
            -
            y_pred
        )
    )

    rmse = np.sqrt(
        np.mean(
            (
                y
                -
                y_pred
            ) ** 2
        )
    )

    print()
    print("===================================")
    print("ОШИБКА АППРОКСИМАЦИИ")
    print("===================================")

    print(
        f"MAE  = {mae:.6f}"
    )

    print(
        f"RMSE = {rmse:.6f}"
    )

    # ========================================================
    # SAVE EXTENDED DATA
    # ========================================================

    output_csv = os.path.join(
        OUTPUT_DIR,
        "gesnr_analysis.csv"
    )

    df.to_csv(
        output_csv,
        index=False
    )

    # ========================================================
    # PLOT 1
    # DATA COLLAPSE
    # ========================================================

    plt.figure(
        figsize=(9, 6)
    )

    graph_types = sorted(
        df[
            "graph_type"
        ].unique()
    )

    for graph_type in graph_types:

        part = df[
            df["graph_type"]
            ==
            graph_type
        ]

        plt.scatter(
            part["gesnr"],
            part["pd_mean"],
            alpha=0.65,
            label=graph_type
        )

    r_min = max(
        df["gesnr"].min(),
        1e-4
    )

    r_max = df[
        "gesnr"
    ].max()

    r_grid = np.logspace(
        np.log10(r_min),
        np.log10(r_max),
        500
    )

    pd_grid = logistic_log_r(
        r_grid,
        a_fit,
        b_fit
    )

    plt.plot(
        r_grid,
        pd_grid,
        linewidth=2.5,
        label=(
            f"Logistic fit, "
            f"R²={r2:.3f}"
        )
    )

    plt.xscale(
        "log"
    )

    plt.xlabel(
        "GESNR"
    )

    plt.ylabel(
        "Detection probability $P_D$"
    )

    plt.title(
        "Data collapse: "
        "$P_D$ vs Graph-Energy SNR"
    )

    plt.legend()

    plt.grid(
        alpha=0.25
    )

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            OUTPUT_DIR,
            "pd_vs_gesnr.png"
        ),
        dpi=300
    )

    plt.close()

    # ========================================================
    # PLOT 2
    # OBSERVED vs PREDICTED
    # ========================================================

    plt.figure(
        figsize=(7, 7)
    )

    plt.scatter(
        df["pd_predicted"],
        df["pd_mean"],
        alpha=0.65
    )

    plt.plot(
        [0, 1],
        [0, 1],
        linestyle="--"
    )

    plt.xlabel(
        "Predicted $P_D$"
    )

    plt.ylabel(
        "Observed $P_D$"
    )

    plt.title(
        "Observed vs predicted "
        "detection probability"
    )

    plt.grid(
        alpha=0.25
    )

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            OUTPUT_DIR,
            "observed_vs_predicted.png"
        ),
        dpi=300
    )

    plt.close()

    # ========================================================
    # PLOT 3
    # RESIDUAL ANALYSIS
    # ========================================================

    plt.figure(
        figsize=(9, 6)
    )

    for graph_type in graph_types:

        part = df[
            df["graph_type"]
            ==
            graph_type
        ]

        plt.scatter(
            part["gesnr"],
            part["residual"],
            alpha=0.65,
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
        "Residual "
        "$P_D - \\hat{P}_D$"
    )

    plt.title(
        "Residual analysis"
    )

    plt.legend()

    plt.grid(
        alpha=0.25
    )

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            OUTPUT_DIR,
            "gesnr_residuals.png"
        ),
        dpi=300
    )

    plt.close()

    # ========================================================
    # PLOT 4
    # UNIVERSAL DETECTION CURVE
    # ========================================================

    plt.figure(
        figsize=(9, 6)
    )

    for graph_type in graph_types:

        part = df[
            df["graph_type"]
            ==
            graph_type
        ]

        plt.scatter(
            part["log_gesnr"],
            part["pd_mean"],
            alpha=0.60,
            label=graph_type
        )

    log_grid = np.linspace(
        df["log_gesnr"].min(),
        df["log_gesnr"].max(),
        500
    )

    r_from_log = np.exp(
        log_grid
    )

    plt.plot(
        log_grid,
        logistic_log_r(
            r_from_log,
            a_fit,
            b_fit
        ),
        linewidth=2.5,
        label="Logistic model"
    )

    plt.xlabel(
        "$\\ln(\\mathrm{GESNR})$"
    )

    plt.ylabel(
        "$P_D$"
    )

    plt.title(
        "Universal detection curve"
    )

    plt.legend()

    plt.grid(
        alpha=0.25
    )

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            OUTPUT_DIR,
            "universal_detection_curve.png"
        ),
        dpi=300
    )

    plt.close()

    # ========================================================
    # PLOT 5
    # ABSOLUTE RESIDUAL
    # ========================================================

    plt.figure(
        figsize=(9, 6)
    )

    for graph_type in graph_types:

        part = df[
            df["graph_type"]
            ==
            graph_type
        ]

        plt.scatter(
            part["gesnr"],
            part["abs_residual"],
            alpha=0.65,
            label=graph_type
        )

    plt.xscale(
        "log"
    )

    plt.xlabel(
        "GESNR"
    )

    plt.ylabel(
        "Absolute prediction error"
    )

    plt.title(
        "Absolute prediction error vs GESNR"
    )

    plt.legend()

    plt.grid(
        alpha=0.25
    )

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            OUTPUT_DIR,
            "absolute_residuals.png"
        ),
        dpi=300
    )

    plt.close()

    # ========================================================
    # TOPOLOGY RESIDUAL STATISTICS
    # ========================================================

    topology_stats = (
        df.groupby(
            "graph_type"
        )["residual"]
        .agg(
            [
                "mean",
                "std",
                "min",
                "max"
            ]
        )
    )

    print()
    print("===================================")
    print("ОСТАТКИ ПО ТОПОЛОГИЯМ")
    print("===================================")

    print(
        topology_stats
    )

    topology_stats.to_csv(
        os.path.join(
            OUTPUT_DIR,
            "residuals_by_topology.csv"
        )
    )

    # ========================================================
    # SAVE MODEL PARAMETERS
    # ========================================================

    model_results = pd.DataFrame(
        {
            "parameter": [
                "pearson_r",
                "pearson_p",
                "spearman_rho",
                "spearman_p",
                "logistic_a",
                "logistic_b",
                "R_squared",
                "MAE",
                "RMSE",
                "GESNR_50",
                "GESNR_90",
                "GESNR_95"
            ],

            "value": [
                pearson_r,
                pearson_p,
                spearman_r,
                spearman_p,
                a_fit,
                b_fit,
                r2,
                mae,
                rmse,
                r50,
                r90,
                r95
            ]
        }
    )

    model_results.to_csv(
        os.path.join(
            OUTPUT_DIR,
            "model_parameters.csv"
        ),
        index=False
    )

    # ========================================================
    # FINISH
    # ========================================================

    print()
    print("===================================")
    print("АНАЛИЗ ЗАВЕРШЁН")
    print("===================================")

    print(
        f"Результаты сохранены в: "
        f"{OUTPUT_DIR}/"
    )

    print()
    print(
        "Созданы файлы:"
    )

    print(
        "  gesnr_analysis.csv"
    )

    print(
        "  model_parameters.csv"
    )

    print(
        "  residuals_by_topology.csv"
    )

    print(
        "  pd_vs_gesnr.png"
    )

    print(
        "  observed_vs_predicted.png"
    )

    print(
        "  gesnr_residuals.png"
    )

    print(
        "  universal_detection_curve.png"
    )

    print(
        "  absolute_residuals.png"
    )


if __name__ == "__main__":
    main()
