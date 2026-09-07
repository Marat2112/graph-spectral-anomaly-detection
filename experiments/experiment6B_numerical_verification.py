import os
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.optimize import minimize, minimize_scalar
from scipy.special import expit


# ============================================================
# EXPERIMENT 6B
# NUMERICAL VERIFICATION OF BINOMIAL MLE
# ============================================================


INPUT_FILE = os.path.join(
    "results",
    "experiment_06",
    "train_grouped.csv"
)

OUTPUT_DIR = "results/experiment_06b"

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


CRITERIA = [
    "R_mean",
    "R_local",
    "R_full",
]


METHODS = [
    "BFGS",
    "L-BFGS-B",
    "Nelder-Mead",
    "Powell",
]


# ============================================================
# NUMERICAL SETTINGS
# ============================================================

EPS = 1e-12

FINITE_DIFF_STEP = 1e-4

PROFILE_POINTS = 121

PROFILE_WIDTH_A = 1.0
PROFILE_WIDTH_B = 1.0

PROFILE_OPT_WIDTH = 15.0

NLL_EQUIVALENCE_TOL = 1e-6


# ============================================================
# LOAD DATA
# ============================================================

def load_data():

    if not os.path.exists(INPUT_FILE):

        raise FileNotFoundError(
            f"\nНе найден файл:\n{INPUT_FILE}\n"
            "\nСначала должен существовать результат "
            "Experiment 6: train_grouped.csv"
        )

    df = pd.read_csv(
        INPUT_FILE
    )

    required_columns = [
        "successes",
        "trials",
        "R_mean",
        "R_local",
        "R_full",
    ]

    missing = [
        col
        for col in required_columns
        if col not in df.columns
    ]

    if missing:

        raise ValueError(
            "Отсутствуют необходимые столбцы: "
            + ", ".join(missing)
        )

    return df


# ============================================================
# STABLE LOGISTIC MODEL
# ============================================================

def logistic_probability(
    R,
    a,
    b
):

    R = np.asarray(
        R,
        dtype=float
    )

    R = np.maximum(
        R,
        EPS
    )

    z = (
        a
        *
        np.log(R)
        +
        b
    )

    return expit(
        z
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

    p = np.clip(
        p,
        EPS,
        1.0 - EPS
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
        np.log1p(-p)
    )

    return float(
        nll
    )


# ============================================================
# ANALYTIC GRADIENT
# ============================================================

def binomial_gradient(
    parameters,
    R,
    successes,
    trials
):

    a, b = parameters

    R = np.maximum(
        np.asarray(
            R,
            dtype=float
        ),
        EPS
    )

    log_R = np.log(
        R
    )

    p = logistic_probability(
        R,
        a,
        b
    )

    residual = (
        trials * p
        -
        successes
    )

    grad_a = np.sum(
        residual
        *
        log_R
    )

    grad_b = np.sum(
        residual
    )

    return np.array(
        [
            grad_a,
            grad_b
        ],
        dtype=float
    )


# ============================================================
# ANALYTIC HESSIAN
# ============================================================

def binomial_hessian(
    parameters,
    R,
    successes,
    trials
):

    a, b = parameters

    R = np.maximum(
        np.asarray(
            R,
            dtype=float
        ),
        EPS
    )

    log_R = np.log(
        R
    )

    p = logistic_probability(
        R,
        a,
        b
    )

    weights = (
        trials
        *
        p
        *
        (
            1.0 - p
        )
    )

    h_aa = np.sum(
        weights
        *
        log_R ** 2
    )

    h_ab = np.sum(
        weights
        *
        log_R
    )

    h_bb = np.sum(
        weights
    )

    H = np.array(
        [
            [
                h_aa,
                h_ab
            ],
            [
                h_ab,
                h_bb
            ]
        ],
        dtype=float
    )

    return H


# ============================================================
# FINITE DIFFERENCE GRADIENT
# ============================================================

def finite_difference_gradient(
    parameters,
    R,
    successes,
    trials,
    h=FINITE_DIFF_STEP
):

    parameters = np.asarray(
        parameters,
        dtype=float
    )

    gradient = np.zeros(
        2,
        dtype=float
    )

    for j in range(2):

        plus = parameters.copy()
        minus = parameters.copy()

        plus[j] += h
        minus[j] -= h

        f_plus = binomial_nll(
            plus,
            R,
            successes,
            trials
        )

        f_minus = binomial_nll(
            minus,
            R,
            successes,
            trials
        )

        gradient[j] = (
            f_plus
            -
            f_minus
        ) / (
            2.0 * h
        )

    return gradient


# ============================================================
# FIT USING ONE OPTIMIZER
# ============================================================

def fit_with_method(
    method,
    R,
    successes,
    trials,
    initial
):

    kwargs = {
        "fun":
            binomial_nll,

        "x0":
            np.asarray(
                initial,
                dtype=float
            ),

        "args":
            (
                R,
                successes,
                trials
            ),

        "method":
            method,
    }

    if method in [
        "BFGS",
        "L-BFGS-B",
    ]:

        kwargs[
            "jac"
        ] = binomial_gradient

    if method == "BFGS":

        kwargs[
            "options"
        ] = {
            "gtol": 1e-6,
            "maxiter": 5000,
        }

    elif method == "L-BFGS-B":

        kwargs[
            "options"
        ] = {
            "ftol": 1e-15,
            "gtol": 1e-8,
            "maxiter": 5000,
            "maxls": 100,
        }

    elif method == "Nelder-Mead":

        kwargs[
            "options"
        ] = {
            "xatol": 1e-10,
            "fatol": 1e-8,
            "maxiter": 20000,
        }

    elif method == "Powell":

        kwargs[
            "options"
        ] = {
            "xtol": 1e-10,
            "ftol": 1e-12,
            "maxiter": 20000,
        }

    with warnings.catch_warnings():

        warnings.simplefilter(
            "ignore"
        )

        result = minimize(
            **kwargs
        )

    parameters = np.asarray(
        result.x,
        dtype=float
    )

    nll = binomial_nll(
        parameters,
        R,
        successes,
        trials
    )

    gradient = binomial_gradient(
        parameters,
        R,
        successes,
        trials
    )

    gradient_norm = float(
        np.linalg.norm(
            gradient
        )
    )

    H = binomial_hessian(
        parameters,
        R,
        successes,
        trials
    )

    eigenvalues = np.linalg.eigvalsh(
        H
    )

    min_hessian_eigenvalue = float(
        np.min(
            eigenvalues
        )
    )

    max_hessian_eigenvalue = float(
        np.max(
            eigenvalues
        )
    )

    hessian_positive_definite = bool(
        np.all(
            eigenvalues > 0
        )
    )

    total_trials = float(
        np.sum(
            trials
        )
    )

    return {

        "method":
            method,

        "success":
            bool(
                result.success
            ),

        "a":
            float(
                parameters[0]
            ),

        "b":
            float(
                parameters[1]
            ),

        "NLL":
            nll,

        "NLL_per_trial":
            nll
            /
            total_trials,

        "gradient_a":
            float(
                gradient[0]
            ),

        "gradient_b":
            float(
                gradient[1]
            ),

        "gradient_norm":
            gradient_norm,

        "hessian_min_eigenvalue":
            min_hessian_eigenvalue,

        "hessian_max_eigenvalue":
            max_hessian_eigenvalue,

        "hessian_positive_definite":
            hessian_positive_definite,

        "iterations":
            getattr(
                result,
                "nit",
                np.nan
            ),

        "function_evaluations":
            getattr(
                result,
                "nfev",
                np.nan
            ),

        "message":
            str(
                result.message
            )
    }


# ============================================================
# MULTI-START CHECK
# ============================================================

def multistart_check(
    R,
    successes,
    trials
):

    starting_points = [
        [-5.0, -5.0],
        [-2.0, 2.0],
        [0.0, 0.0],
        [1.0, -1.0],
        [2.0, -2.0],
        [3.0, -3.0],
        [5.0, -5.0],
        [8.0, -8.0],
        [10.0, 0.0],
        [1.0, 5.0],
    ]

    rows = []

    for start_id, initial in enumerate(
        starting_points
    ):

        result = fit_with_method(
            method="L-BFGS-B",
            R=R,
            successes=successes,
            trials=trials,
            initial=initial
        )

        result[
            "start_id"
        ] = start_id

        result[
            "initial_a"
        ] = initial[0]

        result[
            "initial_b"
        ] = initial[1]

        rows.append(
            result
        )

    return pd.DataFrame(
        rows
    )


# ============================================================
# CHOOSE FINAL MLE
# ============================================================

def choose_final_mle(
    method_df,
    multistart_df
):

    all_candidates = []

    for _, row in method_df.iterrows():

        all_candidates.append(
            {
                "source":
                    "optimizer",

                "source_name":
                    row["method"],

                "a":
                    float(
                        row["a"]
                    ),

                "b":
                    float(
                        row["b"]
                    ),

                "NLL":
                    float(
                        row["NLL"]
                    ),

                "gradient_norm":
                    float(
                        row[
                            "gradient_norm"
                        ]
                    )
            }
        )

    for _, row in multistart_df.iterrows():

        all_candidates.append(
            {
                "source":
                    "multistart",

                "source_name":
                    f"start_{int(row['start_id'])}",

                "a":
                    float(
                        row["a"]
                    ),

                "b":
                    float(
                        row["b"]
                    ),

                "NLL":
                    float(
                        row["NLL"]
                    ),

                "gradient_norm":
                    float(
                        row[
                            "gradient_norm"
                        ]
                    )
            }
        )

    candidate_df = pd.DataFrame(
        all_candidates
    )

    minimum_nll = float(
        candidate_df[
            "NLL"
        ].min()
    )

    near_optimal = candidate_df[
        candidate_df[
            "NLL"
        ]
        <=
        minimum_nll
        +
        NLL_EQUIVALENCE_TOL
    ].copy()

    best_index = near_optimal[
        "gradient_norm"
    ].idxmin()

    best = near_optimal.loc[
        best_index
    ]

    return (
        best,
        candidate_df
    )


# ============================================================
# PROFILE LIKELIHOOD FOR a
# ============================================================

def profile_a(
    R,
    successes,
    trials,
    a_hat,
    b_hat
):

    a_grid = np.linspace(
        a_hat - PROFILE_WIDTH_A,
        a_hat + PROFILE_WIDTH_A,
        PROFILE_POINTS
    )

    rows = []

    b_lower = (
        b_hat
        -
        PROFILE_OPT_WIDTH
    )

    b_upper = (
        b_hat
        +
        PROFILE_OPT_WIDTH
    )

    for a_value in a_grid:

        def objective(
            b_value
        ):

            return binomial_nll(
                [
                    a_value,
                    b_value
                ],
                R,
                successes,
                trials
            )

        result = minimize_scalar(
            objective,
            bounds=(
                b_lower,
                b_upper
            ),
            method="bounded",
            options={
                "xatol": 1e-10,
                "maxiter": 5000
            }
        )

        rows.append(
            {
                "a":
                    float(
                        a_value
                    ),

                "profile_b":
                    float(
                        result.x
                    ),

                "profile_NLL":
                    float(
                        result.fun
                    ),

                "optimizer_success":
                    bool(
                        result.success
                    )
            }
        )

    return pd.DataFrame(
        rows
    )


# ============================================================
# PROFILE LIKELIHOOD FOR b
# ============================================================

def profile_b(
    R,
    successes,
    trials,
    a_hat,
    b_hat
):

    b_grid = np.linspace(
        b_hat - PROFILE_WIDTH_B,
        b_hat + PROFILE_WIDTH_B,
        PROFILE_POINTS
    )

    rows = []

    a_lower = (
        a_hat
        -
        PROFILE_OPT_WIDTH
    )

    a_upper = (
        a_hat
        +
        PROFILE_OPT_WIDTH
    )

    for b_value in b_grid:

        def objective(
            a_value
        ):

            return binomial_nll(
                [
                    a_value,
                    b_value
                ],
                R,
                successes,
                trials
            )

        result = minimize_scalar(
            objective,
            bounds=(
                a_lower,
                a_upper
            ),
            method="bounded",
            options={
                "xatol": 1e-10,
                "maxiter": 5000
            }
        )

        rows.append(
            {
                "b":
                    float(
                        b_value
                    ),

                "profile_a":
                    float(
                        result.x
                    ),

                "profile_NLL":
                    float(
                        result.fun
                    ),

                "optimizer_success":
                    bool(
                        result.success
                    )
            }
        )

    return pd.DataFrame(
        rows
    )


# ============================================================
# COVARIANCE / STANDARD ERRORS
# ============================================================

def covariance_from_hessian(
    parameters,
    R,
    successes,
    trials
):

    H = binomial_hessian(
        parameters,
        R,
        successes,
        trials
    )

    covariance = np.linalg.inv(
        H
    )

    se = np.sqrt(
        np.diag(
            covariance
        )
    )

    correlation = (
        covariance[0, 1]
        /
        (
            se[0]
            *
            se[1]
        )
    )

    return (
        covariance,
        se,
        correlation
    )


# ============================================================
# VERIFY ONE CRITERION
# ============================================================

def verify_criterion(
    df,
    criterion
):

    print()
    print(
        "============================================"
    )

    print(
        f"VERIFYING: {criterion}"
    )

    print(
        "============================================"
    )

    R = df[
        criterion
    ].to_numpy(
        dtype=float
    )

    successes = df[
        "successes"
    ].to_numpy(
        dtype=float
    )

    trials = df[
        "trials"
    ].to_numpy(
        dtype=float
    )

    # ========================================================
    # 1. FOUR OPTIMIZERS
    # ========================================================

    method_rows = []

    for method in METHODS:

        result = fit_with_method(
            method=method,
            R=R,
            successes=successes,
            trials=trials,
            initial=[
                2.0,
                -2.0
            ]
        )

        result[
            "criterion"
        ] = criterion

        method_rows.append(
            result
        )

    method_df = pd.DataFrame(
        method_rows
    )

    print()
    print(
        "OPTIMIZER COMPARISON"
    )

    print(
        method_df[
            [
                "method",
                "success",
                "a",
                "b",
                "NLL",
                "NLL_per_trial",
                "gradient_norm",
                "hessian_min_eigenvalue",
                "hessian_positive_definite",
            ]
        ].to_string(
            index=False
        )
    )

    # ========================================================
    # 2. MULTI-START
    # ========================================================

    multistart_df = multistart_check(
        R,
        successes,
        trials
    )

    multistart_df[
        "criterion"
    ] = criterion

    print()
    print(
        "MULTI-START L-BFGS-B"
    )

    print(
        multistart_df[
            [
                "start_id",
                "initial_a",
                "initial_b",
                "success",
                "a",
                "b",
                "NLL",
                "gradient_norm",
            ]
        ].to_string(
            index=False
        )
    )

    # ========================================================
    # 3. FINAL MLE
    # ========================================================

    best, candidate_df = choose_final_mle(
        method_df,
        multistart_df
    )

    a_hat = float(
        best[
            "a"
        ]
    )

    b_hat = float(
        best[
            "b"
        ]
    )

    best_parameters = np.array(
        [
            a_hat,
            b_hat
        ],
        dtype=float
    )

    min_nll = binomial_nll(
        best_parameters,
        R,
        successes,
        trials
    )

    total_trials = float(
        np.sum(
            trials
        )
    )

    print()
    print(
        "FINAL MLE SOURCE"
    )

    print(
        f"source      = "
        f"{best['source']}"
    )

    print(
        f"source_name = "
        f"{best['source_name']}"
    )

    # ========================================================
    # 4. GRADIENT CHECK
    # ========================================================

    analytic_gradient = binomial_gradient(
        best_parameters,
        R,
        successes,
        trials
    )

    numeric_gradient = finite_difference_gradient(
        best_parameters,
        R,
        successes,
        trials
    )

    gradient_difference = (
        analytic_gradient
        -
        numeric_gradient
    )

    analytic_gradient_norm = float(
        np.linalg.norm(
            analytic_gradient
        )
    )

    finite_diff_gradient_norm = float(
        np.linalg.norm(
            numeric_gradient
        )
    )

    gradient_difference_norm = float(
        np.linalg.norm(
            gradient_difference
        )
    )

    # ========================================================
    # 5. HESSIAN
    # ========================================================

    H = binomial_hessian(
        best_parameters,
        R,
        successes,
        trials
    )

    eigenvalues = np.linalg.eigvalsh(
        H
    )

    hessian_positive_definite = bool(
        np.all(
            eigenvalues > 0
        )
    )

    # ========================================================
    # 6. STANDARD ERRORS
    # ========================================================

    (
        covariance,
        se,
        parameter_corr
    ) = covariance_from_hessian(
        best_parameters,
        R,
        successes,
        trials
    )

    se_a = float(
        se[0]
    )

    se_b = float(
        se[1]
    )

    ci_a_low = (
        a_hat
        -
        1.96
        *
        se_a
    )

    ci_a_high = (
        a_hat
        +
        1.96
        *
        se_a
    )

    ci_b_low = (
        b_hat
        -
        1.96
        *
        se_b
    )

    ci_b_high = (
        b_hat
        +
        1.96
        *
        se_b
    )

    print()
    print(
        "BEST SOLUTION"
    )

    print(
        f"a_hat = "
        f"{a_hat:.12f}"
    )

    print(
        f"b_hat = "
        f"{b_hat:.12f}"
    )

    print(
        f"NLL = "
        f"{min_nll:.12f}"
    )

    print(
        f"NLL per trial = "
        f"{min_nll / total_trials:.12f}"
    )

    print(
        f"SE(a) = "
        f"{se_a:.12f}"
    )

    print(
        f"SE(b) = "
        f"{se_b:.12f}"
    )

    print(
        f"95% CI a = "
        f"[{ci_a_low:.12f}, "
        f"{ci_a_high:.12f}]"
    )

    print(
        f"95% CI b = "
        f"[{ci_b_low:.12f}, "
        f"{ci_b_high:.12f}]"
    )

    print(
        f"corr(a,b) = "
        f"{parameter_corr:.12f}"
    )

    print()
    print(
        "GRADIENT CHECK"
    )

    print(
        "Analytic gradient:",
        analytic_gradient
    )

    print(
        "Finite-difference gradient:",
        numeric_gradient
    )

    print(
        "Difference:",
        gradient_difference
    )

    print(
        f"Analytic gradient norm = "
        f"{analytic_gradient_norm:.12e}"
    )

    print(
        f"Finite diff gradient norm = "
        f"{finite_diff_gradient_norm:.12e}"
    )

    print()
    print(
        "HESSIAN"
    )

    print(
        H
    )

    print()
    print(
        "HESSIAN EIGENVALUES"
    )

    print(
        eigenvalues
    )

    print(
        "Positive definite:",
        hessian_positive_definite
    )

    # ========================================================
    # 7. PROFILE LIKELIHOOD
    # ========================================================

    print()
    print(
        "BUILDING PROFILE LIKELIHOOD..."
    )

    profile_a_df = profile_a(
        R,
        successes,
        trials,
        a_hat,
        b_hat
    )

    profile_b_df = profile_b(
        R,
        successes,
        trials,
        a_hat,
        b_hat
    )

    profile_a_df[
        "delta_NLL"
    ] = (
        profile_a_df[
            "profile_NLL"
        ]
        -
        min_nll
    )

    profile_b_df[
        "delta_NLL"
    ] = (
        profile_b_df[
            "profile_NLL"
        ]
        -
        min_nll
    )

    # ========================================================
    # 8. SAVE TABLES
    # ========================================================

    method_df.to_csv(
        os.path.join(
            OUTPUT_DIR,
            f"{criterion}_optimizer_comparison.csv"
        ),
        index=False
    )

    multistart_df.to_csv(
        os.path.join(
            OUTPUT_DIR,
            f"{criterion}_multistart.csv"
        ),
        index=False
    )

    candidate_df.to_csv(
        os.path.join(
            OUTPUT_DIR,
            f"{criterion}_all_candidates.csv"
        ),
        index=False
    )

    profile_a_df.to_csv(
        os.path.join(
            OUTPUT_DIR,
            f"{criterion}_profile_a.csv"
        ),
        index=False
    )

    profile_b_df.to_csv(
        os.path.join(
            OUTPUT_DIR,
            f"{criterion}_profile_b.csv"
        ),
        index=False
    )

    # ========================================================
    # 9. PROFILE PLOTS
    # ========================================================

    plt.figure(
        figsize=(8, 6)
    )

    plt.plot(
        profile_a_df[
            "a"
        ],
        profile_a_df[
            "delta_NLL"
        ]
    )

    plt.axhline(
        1.92,
        linestyle="--"
    )

    plt.axvline(
        a_hat,
        linestyle="--"
    )

    plt.xlabel(
        "a"
    )

    plt.ylabel(
        "Profile ΔNLL"
    )

    plt.title(
        f"{criterion}: profile likelihood for a"
    )

    plt.grid(
        alpha=0.25
    )

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            OUTPUT_DIR,
            f"{criterion}_profile_a.png"
        ),
        dpi=300
    )

    plt.close()

    plt.figure(
        figsize=(8, 6)
    )

    plt.plot(
        profile_b_df[
            "b"
        ],
        profile_b_df[
            "delta_NLL"
        ]
    )

    plt.axhline(
        1.92,
        linestyle="--"
    )

    plt.axvline(
        b_hat,
        linestyle="--"
    )

    plt.xlabel(
        "b"
    )

    plt.ylabel(
        "Profile ΔNLL"
    )

    plt.title(
        f"{criterion}: profile likelihood for b"
    )

    plt.grid(
        alpha=0.25
    )

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            OUTPUT_DIR,
            f"{criterion}_profile_b.png"
        ),
        dpi=300
    )

    plt.close()

    # ========================================================
    # 10. NUMERICAL SPREADS
    # ========================================================

    optimizer_spread_a = float(
        method_df[
            "a"
        ].max()
        -
        method_df[
            "a"
        ].min()
    )

    optimizer_spread_b = float(
        method_df[
            "b"
        ].max()
        -
        method_df[
            "b"
        ].min()
    )

    optimizer_spread_nll = float(
        method_df[
            "NLL"
        ].max()
        -
        method_df[
            "NLL"
        ].min()
    )

    multistart_spread_a = float(
        multistart_df[
            "a"
        ].max()
        -
        multistart_df[
            "a"
        ].min()
    )

    multistart_spread_b = float(
        multistart_df[
            "b"
        ].max()
        -
        multistart_df[
            "b"
        ].min()
    )

    multistart_spread_nll = float(
        multistart_df[
            "NLL"
        ].max()
        -
        multistart_df[
            "NLL"
        ].min()
    )

    # ========================================================
    # SUMMARY
    # ========================================================

    summary = {

        "criterion":
            criterion,

        "final_source":
            best[
                "source"
            ],

        "final_source_name":
            best[
                "source_name"
            ],

        "a_hat":
            a_hat,

        "b_hat":
            b_hat,

        "SE_a":
            se_a,

        "SE_b":
            se_b,

        "CI95_a_low":
            ci_a_low,

        "CI95_a_high":
            ci_a_high,

        "CI95_b_low":
            ci_b_low,

        "CI95_b_high":
            ci_b_high,

        "parameter_correlation":
            float(
                parameter_corr
            ),

        "best_NLL":
            min_nll,

        "best_NLL_per_trial":
            min_nll
            /
            total_trials,

        "analytic_gradient_norm":
            analytic_gradient_norm,

        "finite_diff_gradient_norm":
            finite_diff_gradient_norm,

        "gradient_difference_norm":
            gradient_difference_norm,

        "hessian_min_eigenvalue":
            float(
                np.min(
                    eigenvalues
                )
            ),

        "hessian_max_eigenvalue":
            float(
                np.max(
                    eigenvalues
                )
            ),

        "hessian_positive_definite":
            hessian_positive_definite,

        "optimizer_spread_a":
            optimizer_spread_a,

        "optimizer_spread_b":
            optimizer_spread_b,

        "optimizer_spread_NLL":
            optimizer_spread_nll,

        "multistart_spread_a":
            multistart_spread_a,

        "multistart_spread_b":
            multistart_spread_b,

        "multistart_spread_NLL":
            multistart_spread_nll,

        "profile_a_all_success":
            bool(
                profile_a_df[
                    "optimizer_success"
                ].all()
            ),

        "profile_b_all_success":
            bool(
                profile_b_df[
                    "optimizer_success"
                ].all()
            ),
    }

    return (
        summary,
        method_df,
        multistart_df,
        candidate_df
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print(
        "============================================"
    )

    print(
        "EXPERIMENT 6B"
    )

    print(
        "NUMERICAL MLE VERIFICATION"
    )

    print(
        "============================================"
    )

    df = load_data()

    print()
    print(
        f"Rows: {len(df)}"
    )

    print(
        f"Total binomial trials: "
        f"{df['trials'].sum():,.0f}"
    )

    all_summaries = []

    all_method_results = []

    all_multistart_results = []

    all_candidates = []

    for criterion in CRITERIA:

        (
            summary,
            method_df,
            multistart_df,
            candidate_df
        ) = verify_criterion(
            df,
            criterion
        )

        all_summaries.append(
            summary
        )

        all_method_results.append(
            method_df
        )

        all_multistart_results.append(
            multistart_df
        )

        candidate_df[
            "criterion"
        ] = criterion

        all_candidates.append(
            candidate_df
        )

    # ========================================================
    # COMBINED RESULTS
    # ========================================================

    summary_df = pd.DataFrame(
        all_summaries
    )

    method_results_df = pd.concat(
        all_method_results,
        ignore_index=True
    )

    multistart_results_df = pd.concat(
        all_multistart_results,
        ignore_index=True
    )

    candidates_df = pd.concat(
        all_candidates,
        ignore_index=True
    )

    summary_df.to_csv(
        os.path.join(
            OUTPUT_DIR,
            "numerical_verification_summary.csv"
        ),
        index=False
    )

    method_results_df.to_csv(
        os.path.join(
            OUTPUT_DIR,
            "all_optimizer_results.csv"
        ),
        index=False
    )

    multistart_results_df.to_csv(
        os.path.join(
            OUTPUT_DIR,
            "all_multistart_results.csv"
        ),
        index=False
    )

    candidates_df.to_csv(
        os.path.join(
            OUTPUT_DIR,
            "all_candidate_solutions.csv"
        ),
        index=False
    )

    # ========================================================
    # FINAL TABLE
    # ========================================================

    print()
    print(
        "============================================"
    )

    print(
        "FINAL NUMERICAL VERIFICATION"
    )

    print(
        "============================================"
    )

    columns_to_show = [
        "criterion",
        "final_source",
        "final_source_name",
        "a_hat",
        "b_hat",
        "SE_a",
        "SE_b",
        "best_NLL_per_trial",
        "analytic_gradient_norm",
        "hessian_min_eigenvalue",
        "hessian_positive_definite",
        "optimizer_spread_a",
        "optimizer_spread_b",
        "optimizer_spread_NLL",
        "multistart_spread_a",
        "multistart_spread_b",
        "multistart_spread_NLL",
        "profile_a_all_success",
        "profile_b_all_success",
    ]

    print(
        summary_df[
            columns_to_show
        ].to_string(
            index=False
        )
    )

    print()
    print(
        "============================================"
    )

    print(
        "EXPERIMENT 6B COMPLETED"
    )

    print(
        "============================================"
    )

    print()

    print(
        f"Results saved to: {OUTPUT_DIR}/"
    )


if __name__ == "__main__":
    main()
