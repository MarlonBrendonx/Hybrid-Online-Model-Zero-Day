import time
from collections import defaultdict, deque

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report

from src.config import ExperimentConfig
from src.models.anomaly_detector import build_hst, calibrate_hst_threshold
from src.models.classifier import build_classifier
from src.osr.centroid import CentroidOSR
from src.osr.entropy import EntropyOSR
from src.pipeline.auto_labeler import ConservativeAutoLabeler
from src.pipeline.selector import PrequentialSelector
from src.preprocessing.features import encode_categoricals
from src.preprocessing.scalers import get_sklearn_scaler
from src.utils.logger import log


def run_experiment(
    zero_day_class: int,
    X_raw: pd.DataFrame,
    y_true: np.ndarray,
    numeric_cols: list,
    categ_cols: list,
    le_classes: np.ndarray,
    config: ExperimentConfig,
):
    zd_name = le_classes[zero_day_class]
    prefix = zd_name

    ZERO_DAY_LABEL = config.zero_day_label
    NORMAL_CLASS = config.normal_class
    ZERO_DAY_CLASSES = [zero_day_class]
    KNOWN_CLASSES = [c for c in range(len(le_classes)) if c not in ZERO_DAY_CLASSES]
    labels_eval = KNOWN_CLASSES + [ZERO_DAY_LABEL]
    target_names = [f"class_{c}" for c in KNOWN_CLASSES] + ["zero_day"]

    # ── 1. Split ──────────────────────────────────────────────────────────
    log("Dividindo índices train/test/zero-day…", prefix)
    mask_known = ~np.isin(y_true, ZERO_DAY_CLASSES)
    idx_known, idx_zd = np.where(mask_known)[0], np.where(~mask_known)[0]
    split_point = int(len(idx_known) * config.train_split)
    train_idx, test_idx = idx_known[:split_point], idx_known[split_point:]

    X_train_raw, y_train = X_raw.iloc[train_idx].copy(), y_true[train_idx]
    X_test_raw, y_test = X_raw.iloc[test_idx].copy(), y_true[test_idx]
    X_zd_raw, y_zd = X_raw.iloc[idx_zd].copy(), y_true[idx_zd]

    _, counts = np.unique(y_test, return_counts=True)
    max_test_count = max(counts) if len(counts) > 0 else len(y_test)
    X_zd_raw, y_zd_sample = X_zd_raw.iloc[:max_test_count], y_zd[:max_test_count]

    log(
        f"  train={len(train_idx)} | test={len(test_idx)} | zero-day={len(y_zd_sample)}",
        prefix,
    )

    # ── 2. Categorical encoding ───────────────────────────────────────────
    log("Pré-processando features (OrdinalEncoder)…", prefix)
    X_train_proc, X_test_proc, X_zd_proc = encode_categoricals(
        X_train_raw, X_test_raw, X_zd_raw, numeric_cols, categ_cols
    )

    feature_names = X_train_proc.columns.tolist()
    X_train_arr = X_train_proc.values
    X_test_arr = X_test_proc.values
    X_zd_arr = X_zd_proc.values

    # ── 3. Model initialisation ───────────────────────────────────────────
    log("Inicializando HST, AdaBoost, OSR-A/B, Selector…", prefix)
    hst = build_hst(config)
    clf = build_classifier(config)
    osr_a = CentroidOSR(
        window=config.centroid_osr_window,
        n_candidates=config.centroid_osr_candidates,
    )
    osr_scaler = get_sklearn_scaler(config.scaler)
    osr_b = EntropyOSR(window_size=config.entropy_osr_window)
    selector = PrequentialSelector(
        window_size=config.selector_window_size,
        zero_day_label=ZERO_DAY_LABEL,
        min_samples=config.selector_min_samples,
    )
    auto_labeler = ConservativeAutoLabeler(
        confidence_threshold=config.autolabel_confidence,
        history_window=config.autolabel_history_window,
        min_history_count=config.autolabel_min_history_count,
        zero_day_label=ZERO_DAY_LABEL,
    )

    # ── 4. HST warm-up on normal samples ─────────────────────────────────
    n_normal = int((y_train == NORMAL_CLASS).sum())
    log(f"Warm-up HST com {n_normal} amostras normais…", prefix)
    for x_row in X_train_arr[y_train == NORMAL_CLASS]:
        hst.learn_one({feature_names[j]: x_row[j] for j in range(len(x_row))})

    # ── 5. HST threshold calibration ─────────────────────────────────────
    log(f"Calculando scores HST no treino ({len(X_train_arr)} amostras)…", prefix)
    hst_threshold = calibrate_hst_threshold(
        hst,
        X_train_arr,
        y_train,
        feature_names,
        NORMAL_CLASS,
        config.hst_threshold_candidates,
    )
    log(f"Limiar HST calibrado: {hst_threshold:.6f}", prefix)

    # ── 6. Online training of clf + OSR ──────────────────────────────────
    log(f"Treinando clf + OSR online ({len(X_train_arr)} amostras)…", prefix)
    osr_scaler.fit(X_train_arr[y_train != NORMAL_CLASS])
    for i, x_row in enumerate(X_train_arr):
        xi = {feature_names[j]: x_row[j] for j in range(len(x_row))}
        cls = y_train[i]
        clf.learn_one(xi, cls)
        if cls != NORMAL_CLASS:
            x_sc = osr_scaler.transform(x_row.reshape(1, -1))[0]
            osr_a.learn_one(x_sc, cls)
            if probas := clf.predict_proba_one(xi):
                osr_b.learn_one(probas)

    # ── 7. OSR calibration ────────────────────────────────────────────────
    log("Calibrando OSR-A (CentroidOSR)…", prefix)
    osr_a.calibrate()
    log("Calibrando OSR-B (EntropyOSR)…", prefix)
    osr_b.calibrate()

    # ── 8. Hybrid evaluation ──────────────────────────────────────────────
    X_full = pd.concat(
        [pd.DataFrame(X_test_arr), pd.DataFrame(X_zd_arr)], axis=0
    ).reset_index(drop=True)
    y_full_arr = np.concatenate([y_test, y_zd_sample])
    n_total = len(y_full_arr)

    log(f"Iniciando avaliação híbrida ({n_total} amostras)…", prefix)
    hybrid_true, hybrid_pred, total_ns, n_samples = [], [], 0, 0
    detector_usage = defaultdict(int)
    autolabel_counts = {"accepted": 0, "rejected": 0}
    expert_queue = deque(maxlen=config.expert_queue_size)

    LOG_STEPS = max(1, n_total // 5)

    for i, x_row in enumerate(X_full.values):
        if i > 0 and i % LOG_STEPS == 0:
            pct = 100 * i // n_total
            log(f"  progresso avaliação híbrida: {pct}% ({i}/{n_total})", prefix)

        t0 = time.time_ns()
        xi = {feature_names[j]: x_row[j] for j in range(len(x_row))}
        y_full = y_full_arr[i]
        y_bin = 0 if y_full == NORMAL_CLASS else 1
        predicted_bin = 1 if hst.score_one(xi) > hst_threshold else 0

        if predicted_bin == 1 and y_bin == 1:
            predicted_cls = clf.predict_one(xi)
            probas = clf.predict_proba_one(xi) or {}
            x_sc = osr_scaler.transform(x_row.reshape(1, -1))[0]

            if predicted_cls == NORMAL_CLASS:
                final_pred = pred_a = pred_b = ZERO_DAY_LABEL
                total_ns += time.time_ns() - t0
            else:
                det = selector.select(predicted_cls)
                detector_usage[det] += 1
                pred_a = (
                    ZERO_DAY_LABEL
                    if osr_a.is_zero_day(x_sc, predicted_cls)
                    else predicted_cls
                )
                pred_b = (
                    ZERO_DAY_LABEL
                    if (probas and osr_b.is_zero_day(probas))
                    else predicted_cls
                )
                final_pred = {"A": pred_a, "B": pred_b}[det]
                total_ns += time.time_ns() - t0

                if auto_cls := auto_labeler.evaluate(
                    predicted_cls, pred_a, pred_b, probas
                ):
                    autolabel_counts["accepted"] += 1
                    osr_a.learn_one(x_sc, auto_cls)
                    osr_b.learn_one(probas)
                    clf.learn_one(xi, auto_cls)
                    hst.learn_one(xi)
                else:
                    autolabel_counts["rejected"] += 1

            y_true_mapped = ZERO_DAY_LABEL if y_full in ZERO_DAY_CLASSES else y_full
            hybrid_true.append(y_true_mapped)
            hybrid_pred.append(final_pred)

            if predicted_cls != NORMAL_CLASS:
                expert_queue.append((predicted_cls, y_true_mapped, pred_a, pred_b))
                if len(expert_queue) == config.expert_queue_size:
                    p_p, p_t, p_a, p_b = expert_queue.popleft()
                    selector.update(p_p, p_t, p_a, p_b)
        else:
            total_ns += time.time_ns() - t0
        n_samples += 1

    log(
        f"Avaliação híbrida concluída. AutoLabel aceitos={autolabel_counts['accepted']} "
        f"| rejeitados={autolabel_counts['rejected']} "
        f"| detector_uso={dict(detector_usage)}",
        prefix,
    )

    # ── 9. Hybrid F1 ──────────────────────────────────────────────────────
    log("Calculando F1 híbrido…", prefix)
    f1_hybrid = (
        classification_report(
            hybrid_true,
            hybrid_pred,
            labels=labels_eval,
            target_names=target_names,
            zero_division=0,
            output_dict=True,
        )["zero_day"]["f1-score"]
        if hybrid_true
        else 0.0
    )

    # ── 10. Baseline ──────────────────────────────────────────────────────
    log("Treinando e avaliando baseline AdaBoost…", prefix)
    bl = build_classifier(config)
    for i, x_row in enumerate(X_train_arr):
        bl.learn_one(
            {feature_names[j]: x_row[j] for j in range(len(x_row))}, y_train[i]
        )

    bl_true, bl_pred, bl_ns = [], [], 0
    for i, x_row in enumerate(X_full.values):
        t0 = time.time_ns()
        xi = {feature_names[j]: x_row[j] for j in range(len(x_row))}
        pred = bl.predict_one(xi)
        probas = bl.predict_proba_one(xi)
        fp = (
            ZERO_DAY_LABEL
            if (max(probas.values()) if probas else 0) < 0.70
            else pred
        )
        bl_ns += time.time_ns() - t0
        bl_true.append(
            ZERO_DAY_LABEL if y_full_arr[i] in ZERO_DAY_CLASSES else y_full_arr[i]
        )
        bl_pred.append(fp)

    f1_baseline = classification_report(
        bl_true,
        bl_pred,
        labels=labels_eval,
        target_names=target_names,
        zero_division=0,
        output_dict=True,
    )["zero_day"]["f1-score"]

    log(
        f"CONCLUÍDO — Hybrid_F1={f1_hybrid:.4f} | Baseline_F1={f1_baseline:.4f} "
        f"| Delta={f1_hybrid - f1_baseline:+.4f}",
        prefix,
    )

    return {
        "Zero_Day_Class": zd_name,
        "Hybrid_F1": f1_hybrid,
        "Baseline_F1": f1_baseline,
        "Delta_F1": f1_hybrid - f1_baseline,
        "Hybrid_Latency_us": (total_ns / n_samples) / 1000,
        "Baseline_Latency_us": (bl_ns / n_samples) / 1000,
        "AutoLabel_AcceptRate": autolabel_counts["accepted"]
        / max(autolabel_counts["accepted"] + autolabel_counts["rejected"], 1),
    }
