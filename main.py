import argparse
import concurrent.futures
import os

import numpy as np
import pandas as pd

from src.config import ExperimentConfig
from src.pipeline.experiment import run_experiment
from src.preprocessing.features import (
    drop_constant_columns,
    encode_labels,
    find_normal_class,
    load_dataset,
    preprocess_features,
)
from src.utils.logger import log


def parse_args() -> ExperimentConfig:
    parser = argparse.ArgumentParser(
        description="Hybrid Online Zero-Day Detection",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--scaler",
        choices=["standard", "minmax", "robust"],
        default="standard",
        help="Normalization for HST, clf pipeline and OSR-A scaler",
    )
    parser.add_argument(
        "--dataset",
        default="",
        help="Path to dataset CSV. Auto-detected from cwd if omitted.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=-1,
        help="Parallel workers (-1 = cpu_count - 2)",
    )
    args = parser.parse_args()
    return ExperimentConfig(
        scaler=args.scaler,
        dataset_path=args.dataset,
        n_workers=args.workers,
    )


if __name__ == "__main__":
    config = parse_args()
    log(f"Iniciando experimento | scaler='{config.scaler}'")

    if config.dataset_path:
        path = config.dataset_path
    else:
        path = (
            "./ML_EdgeIIoT_SMOTE.csv"
            if os.path.exists("./ML_EdgeIIoT_SMOTE.csv")
            else "./ERENO-2.0-100K.csv"
        )
    log(f"Carregando dataset: {path}")

    df, target_col = load_dataset(path)
    log(f"Dataset carregado: {len(df)} linhas, {len(df.columns)} colunas.")

    y_raw = df[target_col].astype(str)
    X_raw = df.drop(columns=[target_col])
    numeric_cols = X_raw.select_dtypes(include=[np.number]).columns.tolist()
    categ_cols = X_raw.select_dtypes(exclude=[np.number]).columns.tolist()

    X_raw, numeric_cols = drop_constant_columns(X_raw, numeric_cols)

    log("Aplicando log1p em colunas numéricas skewed…")
    X_raw = preprocess_features(X_raw, numeric_cols)

    y_true, le = encode_labels(y_raw)
    config.normal_class = find_normal_class(le)

    classes_to_test = [c for c in range(len(le.classes_)) if c != config.normal_class]
    log(
        f"Dataset: {path} | Normal class idx={config.normal_class} | "
        f"Classes a testar: {len(classes_to_test)}"
    )
    log(f"Classes: {list(le.classes_)}")

    results = []
    n_workers = config.resolve_workers()
    log(f"Disparando ProcessPoolExecutor com {n_workers} workers…")

    with concurrent.futures.ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = {
            executor.submit(
                run_experiment,
                zd,
                X_raw,
                y_true,
                numeric_cols,
                categ_cols,
                le.classes_,
                config,
            ): zd
            for zd in classes_to_test
        }
        done_count = 0
        for f in concurrent.futures.as_completed(futures):
            try:
                res = f.result()
                results.append(res)
                done_count += 1
                log(
                    f"[{done_count}/{len(classes_to_test)}] Concluído: "
                    f"{res['Zero_Day_Class']} | F1={res['Hybrid_F1']:.4f} | "
                    f"Delta={res['Delta_F1']:+.4f}"
                )
            except Exception as e:
                done_count += 1
                log(f"[{done_count}/{len(classes_to_test)}] ERRO: {e}")

    if results:
        SEP = "=" * 76
        df_r = pd.DataFrame(results).sort_values("Zero_Day_Class")
        log("Todos os experimentos finalizados. Exibindo resumo…")
        print(f"\n{SEP}\nRESUMO FINAL\n{SEP}\n{df_r.to_string(index=False)}\n{SEP}")
        print(
            f"Média F1 Híbrido: {df_r['Hybrid_F1'].mean():.4f} | "
            f"Delta: {df_r['Delta_F1'].mean():+.4f}\n{SEP}"
        )
    else:
        log("Nenhum resultado disponível — verifique os erros acima.")
