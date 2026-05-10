from river import compose, ensemble, tree

from src.config import ExperimentConfig
from src.preprocessing.scalers import get_river_scaler


def build_classifier(config: ExperimentConfig):
    return compose.Pipeline(
        get_river_scaler(config.scaler),
        ensemble.AdaBoostClassifier(
            model=tree.HoeffdingTreeClassifier(
                split_criterion=config.hoeffding_split_criterion,
                max_depth=config.hoeffding_max_depth,
            ),
            n_models=config.adaboost_n_models,
            seed=config.adaboost_seed,
        ),
    )
