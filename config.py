from dataclasses import dataclass, field


@dataclass
class ExperimentConfig:
    # -----------------------------
    # GRAPH
    # -----------------------------
    num_sensors: int = 20
    graph_type: str = "ring"

    # -----------------------------
    # NOISE
    # -----------------------------
    noise_levels: list[float] = field(
        default_factory=lambda: [
            0.05,
            0.10,
            0.15,
            0.20,
            0.25,
            0.30,
            0.40,
            0.50,
        ]
    )

    # -----------------------------
    # DETECTOR
    # -----------------------------
    false_alarm_alpha: float = 0.01

    # Шум, на котором один раз калибруется
    # фиксированный порог
    baseline_noise: float = 0.10

    # -----------------------------
    # ANOMALY
    # -----------------------------
    anomaly_amplitude: float = 1.5

    # Сколько сенсоров одновременно
    # затрагивает локальная аномалия
    anomaly_width: int = 1

    # -----------------------------
    # MONTE CARLO
    # -----------------------------
    background_samples: int = 3000
    test_normal_samples: int = 3000
    test_anomaly_samples: int = 3000

    # Сколько раз полностью повторяем
    # эксперимент независимо
    repetitions: int = 30

    # -----------------------------
    # REPRODUCIBILITY
    # -----------------------------
    random_seed: int = 42

    # -----------------------------
    # OUTPUT
    # -----------------------------
    results_dir: str = "results/experiment_01"
