import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder

from src.utils.logger import log


def load_dataset(path: str) -> tuple[pd.DataFrame, str]:
    df = pd.read_csv(path, low_memory=False)
    target_col = "Attack_type" if "Attack_type" in df.columns else "class"
    df = df.dropna(subset=[target_col])
    return df, target_col


def drop_constant_columns(
    X_raw: pd.DataFrame, numeric_cols: list
) -> tuple[pd.DataFrame, list]:
    constant_cols = [col for col in numeric_cols if X_raw[col].nunique() <= 1]
    if constant_cols:
        log(f"Removendo {len(constant_cols)} colunas constantes.")
        X_raw = X_raw.drop(columns=constant_cols)
        numeric_cols = [c for c in numeric_cols if c not in constant_cols]
    return X_raw, numeric_cols


def preprocess_features(X_raw: pd.DataFrame, numeric_cols: list) -> pd.DataFrame:
    for col in numeric_cols:
        if X_raw[col].min() >= 0 and X_raw[col].max() > 1000:
            X_raw[col] = np.log1p(X_raw[col])
    X_raw = X_raw.replace([np.inf, -np.inf], np.nan).fillna(0)
    X_raw[numeric_cols] = X_raw[numeric_cols].clip(lower=-1e6, upper=1e6)
    return X_raw


def encode_labels(y_raw: pd.Series) -> tuple[np.ndarray, LabelEncoder]:
    le = LabelEncoder()
    y_true = le.fit_transform(y_raw)
    return y_true, le


def find_normal_class(le: LabelEncoder) -> int:
    for i, name in enumerate(le.classes_):
        if name.lower() == "normal":
            return i
    raise ValueError("Classe 'normal' não encontrada no dataset.")


def encode_categoricals(
    X_train_raw: pd.DataFrame,
    X_test_raw: pd.DataFrame,
    X_zd_raw: pd.DataFrame,
    numeric_cols: list,
    categ_cols: list,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if not categ_cols:
        return (
            X_train_raw[numeric_cols],
            X_test_raw[numeric_cols],
            X_zd_raw[numeric_cols],
        )

    oe = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    X_tr_cat = X_train_raw[categ_cols].fillna("##MISSING##").astype(str)
    X_te_cat = X_test_raw[categ_cols].fillna("##MISSING##").astype(str)
    X_z_cat = X_zd_raw[categ_cols].fillna("##MISSING##").astype(str)

    X_train_proc = pd.concat(
        [
            X_train_raw[numeric_cols],
            pd.DataFrame(
                oe.fit_transform(X_tr_cat),
                columns=categ_cols,
                index=X_train_raw.index,
            ),
        ],
        axis=1,
    )
    X_test_proc = pd.concat(
        [
            X_test_raw[numeric_cols],
            pd.DataFrame(
                oe.transform(X_te_cat),
                columns=categ_cols,
                index=X_test_raw.index,
            ),
        ],
        axis=1,
    )
    X_zd_proc = pd.concat(
        [
            X_zd_raw[numeric_cols],
            pd.DataFrame(
                oe.transform(X_z_cat),
                columns=categ_cols,
                index=X_zd_raw.index,
            ),
        ],
        axis=1,
    )
    return X_train_proc, X_test_proc, X_zd_proc
