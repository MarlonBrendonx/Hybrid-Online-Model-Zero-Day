import numpy as np
from river import anomaly, compose
from sklearn.metrics import f1_score

from src.config import ExperimentConfig
from src.preprocessing.scalers import get_river_scaler


def build_hst(config: ExperimentConfig):
    return compose.Pipeline(
        get_river_scaler(config.scaler),
        anomaly.HalfSpaceTrees(
            n_trees=config.hst_n_trees,
            height=config.hst_height,
            window_size=config.hst_window_size,
            seed=config.hst_seed,
        ),
    )


def calibrate_hst_threshold(
    hst,
    X_train_arr: np.ndarray,
    y_train: np.ndarray,
    feature_names: list,
    normal_class: int,
    n_candidates: int,
) -> float:
    scores_train, labels_binary = [], []
    for i, x_row in enumerate(X_train_arr):
        xi = {feature_names[j]: x_row[j] for j in range(len(x_row))}
        scores_train.append(hst.score_one(xi))
        labels_binary.append(0 if y_train[i] == normal_class else 1)

    scores_arr = np.array(scores_train)
    thr_cands = np.linspace(scores_arr.min(), scores_arr.max(), n_candidates)
    return float(
        thr_cands[
            np.argmax(
                [
                    f1_score(
                        np.array(labels_binary),
                        (scores_arr > t).astype(int),
                        zero_division=0,
                    )
                    for t in thr_cands
                ]
            )
        ]
    )
