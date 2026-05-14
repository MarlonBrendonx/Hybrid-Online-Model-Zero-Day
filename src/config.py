import os
from dataclasses import dataclass
from typing import Literal


@dataclass
class ExperimentConfig:
    # ── Scaler ────────────────────────────────────────────────────────────
    # Controls preprocessing for HST pipeline, clf pipeline, and osr_scaler.
    # "robust" maps to RobustScaler for sklearn; River has no RobustScaler
    # so it falls back to StandardScaler there.
    scaler: Literal["standard", "minmax", "robust"] = "standard"

    # ── Dataset ───────────────────────────────────────────────────────────
    dataset_path: str = ""   # auto-detected from cwd if empty
    target_col: str = ""     # auto-detected from column names if empty

    # ── Labels ────────────────────────────────────────────────────────────
    zero_day_label: int = 9
    normal_class: int = -1   # filled after LabelEncoder runs in main

    # ── Split ─────────────────────────────────────────────────────────────
    train_split: float = 0.90

    # ── Execution ─────────────────────────────────────────────────────────
    n_workers: int = -1      # -1 → cpu_count() - 2

    # ── HalfSpaceTrees ────────────────────────────────────────────────────
    hst_n_trees: int = 10
    hst_height: int = 11
    hst_window_size: int = 100
    hst_seed: int = 42
    hst_threshold_candidates: int = 300

    # ── AdaBoost / HoeffdingTree ──────────────────────────────────────────
    adaboost_n_models: int = 15
    adaboost_seed: int = 42
    hoeffding_split_criterion: str = "gini"
    hoeffding_max_depth: int = 15

    # ── CentroidOSR ───────────────────────────────────────────────────────
    centroid_osr_window: int = 2000
    centroid_osr_candidates: int = 400

    # ── EntropyOSR ────────────────────────────────────────────────────────
    entropy_osr_window: int = 2000

    # ── PrequentialSelector ───────────────────────────────────────────────
    selector_window_size: int = 200
    selector_min_samples: int = 30

    # ── ConservativeAutoLabeler ───────────────────────────────────────────
    autolabel_confidence: float = 0.90
    autolabel_history_window: int = 50
    autolabel_min_history_count: int = 3

    # ── Misc ──────────────────────────────────────────────────────────────
    expert_queue_size: int = 100
    baseline_only: bool = False

    def resolve_workers(self) -> int:
        if self.n_workers > 0:
            return self.n_workers
        return max(1, os.cpu_count() - 2)
