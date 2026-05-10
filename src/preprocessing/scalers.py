from river import preprocessing as river_prep
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler

_SKLEARN_MAP = {
    "standard": StandardScaler,
    "minmax": MinMaxScaler,
    "robust": RobustScaler,
}

# River does not have RobustScaler; "robust" falls back to StandardScaler.
_RIVER_MAP = {
    "standard": river_prep.StandardScaler,
    "minmax": river_prep.MinMaxScaler,
    "robust": river_prep.StandardScaler,
}


def get_sklearn_scaler(scaler_type: str):
    if scaler_type not in _SKLEARN_MAP:
        raise ValueError(f"Unknown scaler '{scaler_type}'. Options: {list(_SKLEARN_MAP)}")
    return _SKLEARN_MAP[scaler_type]()


def get_river_scaler(scaler_type: str):
    if scaler_type not in _RIVER_MAP:
        raise ValueError(f"Unknown scaler '{scaler_type}'. Options: {list(_RIVER_MAP)}")
    return _RIVER_MAP[scaler_type]()
