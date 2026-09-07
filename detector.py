import numpy as np
from scipy.stats import norm


def empirical_threshold(
    background_energies: np.ndarray,
    alpha: float
) -> float:
    """
    Эмпирический квантильный порог.

        tau = F_hat^{-1}(1 - alpha)

    alpha — требуемая вероятность
    ложной тревоги.
    """

    if not 0 < alpha < 1:
        raise ValueError(
            "alpha должна принадлежать (0, 1)"
        )

    tau = np.quantile(
        background_energies,
        1.0 - alpha
    )

    return float(tau)


def gaussian_threshold(
    background_energies: np.ndarray,
    alpha: float
) -> float:
    """
    Гауссовское приближение:

        tau = mu + z_(1-alpha) * sigma.
    """

    mu = np.mean(background_energies)

    sigma = np.std(
        background_energies,
        ddof=1
    )

    z = norm.ppf(1.0 - alpha)

    tau = mu + z * sigma

    return float(tau)


def detect(
    energies: np.ndarray,
    tau: float
) -> np.ndarray:
    """
    delta = 1, если Q(x) > tau.
    """

    return energies > tau


def false_alarm_rate(
    normal_energies: np.ndarray,
    tau: float
) -> float:
    """
    P_FA = P(delta=1 | H0)
    """

    alarms = detect(
        normal_energies,
        tau
    )

    return float(
        np.mean(alarms)
    )


def detection_probability(
    anomaly_energies: np.ndarray,
    tau: float
) -> float:
    """
    P_D = P(delta=1 | H1)
    """

    alarms = detect(
        anomaly_energies,
        tau
    )

    return float(
        np.mean(alarms)
    )