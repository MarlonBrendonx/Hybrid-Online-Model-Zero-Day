import argparse
import concurrent.futures
import os

import numpy as np
import pandas as pd

from src.config import ExperimentConfig
from src.pipeline.experiment import run_baseline, run_experiment
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
    parser.add_argument(
        "--baseline-only",
        action="store_true",
        default=False,
        help="Run only the AdaBoost baseline (skip hybrid pipeline).",
    )
    args = parser.parse_args()
    return ExperimentConfig(
        scaler=args.scaler,
        dataset_path=args.dataset,
        n_workers=args.workers,
        baseline_only=args.baseline_only,
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
    task_fn = run_baseline if config.baseline_only else run_experiment
    mode_label = "BASELINE" if config.baseline_only else "HÍBRIDO"
    log(f"Modo: {mode_label} | Disparando ProcessPoolExecutor com {n_workers} workers…")

    with concurrent.futures.ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = {
            executor.submit(
                task_fn,
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
                if config.baseline_only:
                    log(
                        f"[{done_count}/{len(classes_to_test)}] Concluído: "
                        f"{res['Zero_Day_Class']} | Baseline_F1={res['Baseline_F1']:.4f}"
                    )
                else:
                    log(
                        f"[{done_count}/{len(classes_to_test)}] Concluído: "
                        f"{res['Zero_Day_Class']} | F1={res['Hybrid_F1']:.4f} | "
                        f"Delta={res['Delta_F1']:+.4f}"
                    )
            except Exception as e:
                done_count += 1
                log(f"[{done_count}/{len(classes_to_test)}] ERRO: {e}")

    if results and config.baseline_only:
        SEP = "=" * 60
        df_r = pd.DataFrame(results)[["Zero_Day_Class", "Baseline_F1", "Baseline_Latency_us"]].sort_values("Zero_Day_Class")
        log("Todos os experimentos baseline finalizados.")
        print(f"\n{SEP}\nRESUMO BASELINE\n{SEP}\n{df_r.to_string(index=False)}\n{SEP}")
        print(f"Média Baseline_F1: {df_r['Baseline_F1'].mean():.4f}\n{SEP}")

    elif results:
        SEP = "=" * 76

        series_data = {}
        for res in results:
            zd = res["Zero_Day_Class"]
            series_data[zd] = {
                "autolabel_error_series": res.pop("_autolabel_error_series", []),
                "f1_rolling_series": res.pop("_f1_rolling_series", []),
            }

        scalar_cols = [
            "Zero_Day_Class", "Hybrid_F1", "Baseline_F1", "Delta_F1",
            "Hybrid_Latency_us", "Baseline_Latency_us",
            "AutoLabel_AcceptRate", "AutoLabel_ErrorRate",
        ]
        df_r = pd.DataFrame(results)[scalar_cols].sort_values("Zero_Day_Class")
        log("Todos os experimentos finalizados. Exibindo resumo…")
        print(f"\n{SEP}\nRESUMO FINAL\n{SEP}\n{df_r.to_string(index=False)}\n{SEP}")
        print(
            f"Média F1 Híbrido: {df_r['Hybrid_F1'].mean():.4f} | "
            f"Delta: {df_r['Delta_F1'].mean():+.4f}\n{SEP}"
        )

        print(f"\n{SEP}\nANÁLISE TEMPORAL — AutoLabel Error Rate & F1 Rolling\n{SEP}")
        header = f"{'Classe':<28} {'ErrPeak':>8} {'ErrFinal':>9} {'F1_Min':>7} {'F1_Max':>7} {'F1_Std':>7} {'Tendência'}"
        print(header)
        print("-" * len(header))
        for res in sorted(results, key=lambda r: r["Zero_Day_Class"]):
            zd = res["Zero_Day_Class"]
            sd = series_data[zd]
            err_s = sd["autolabel_error_series"]
            f1_s = sd["f1_rolling_series"]

            err_peak = max((e for _, e in err_s), default=0.0)
            err_final = err_s[-1][1] if err_s else 0.0
            f1_vals = [f for _, f in f1_s]
            f1_min = min(f1_vals, default=res["Hybrid_F1"])
            f1_max = max(f1_vals, default=res["Hybrid_F1"])
            f1_std = float(np.std(f1_vals)) if f1_vals else 0.0

            if len(f1_vals) >= 2:
                mid = len(f1_vals) // 2
                first_half = np.mean(f1_vals[:mid])
                second_half = np.mean(f1_vals[mid:])
                delta = second_half - first_half
                if delta < -0.05:
                    trend = "↓ Degradou"
                elif delta > 0.05:
                    trend = "↑ Melhorou"
                else:
                    trend = "→ Estável"
            else:
                trend = "— Poucos pts"

            stability_flag = " [!]" if f1_std > 0.05 or err_peak > 0.30 else ""
            print(
                f"{zd:<28} {err_peak:>8.3f} {err_final:>9.3f} "
                f"{f1_min:>7.4f} {f1_max:>7.4f} {f1_std:>7.4f} "
                f"{trend}{stability_flag}"
            )
        print(SEP)
    else:
        log("Nenhum resultado disponível — verifique os erros acima.")
