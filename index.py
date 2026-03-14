import time
import numpy as np
import pandas as pd
import concurrent.futures
import os
from collections import defaultdict, deque

from river import anomaly, compose, preprocessing, ensemble, tree
from sklearn.preprocessing import (
    LabelEncoder,
    OrdinalEncoder,
    StandardScaler,
    MinMaxScaler,
)
from sklearn.metrics import classification_report, f1_score


def log(msg: str, prefix: str = ""):
    ts = time.strftime("%H:%M:%S")
    tag = f"[{prefix}] " if prefix else ""
    print(f"[{ts}] {tag}{msg}", flush=True)


class ConservativeAutoLabeler:
    def __init__(
        self,
        confidence_threshold: float = 0.90,
        history_window: int = 50,
        min_history_count: int = 3,
        zero_day_label: int = 9,
    ):
        self.confidence_threshold = confidence_threshold
        self.history_window = history_window
        self.min_history_count = min_history_count
        self.ZERO_DAY = zero_day_label
        self._recent_accepted: deque = deque(maxlen=history_window)
        self.stats = {"accepted": 0, "rejected": 0}

    def evaluate(
        self, predicted_cls: int, pred_a: int, pred_b: int, probas: dict
    ) -> int | None:
        predictions = {pred_a, pred_b}
        if len(predictions) != 1:
            self.stats["rejected"] += 1
            return None
        agreed_cls = pred_a

        if agreed_cls == self.ZERO_DAY:
            self.stats["rejected"] += 1
            return None

        max_conf = max(probas.values()) if probas else 0.0
        if max_conf < self.confidence_threshold:
            self.stats["rejected"] += 1
            return None

        if len(self._recent_accepted) >= self.history_window:
            count = sum(1 for c in self._recent_accepted if c == agreed_cls)
            if count < self.min_history_count:
                self.stats["rejected"] += 1
                return None

        self._recent_accepted.append(agreed_cls)
        self.stats["accepted"] += 1
        return agreed_cls


class CentroidOSR:
    def __init__(self):
        self.n: dict = {}
        self.mean: dict = {}
        self.M2: dict = {}
        self.thresholds: dict = {}
        self._intra: dict = {}
        self._inter: dict = {}

    def _dist(self, x: np.ndarray, cls: int) -> float:
        if cls not in self.mean:
            return float("inf")
        var = self.M2[cls] / max(self.n[cls] - 1, 1)
        std = np.sqrt(np.where(var < 1e-9, 1e-9, var))
        return float(np.mean(np.abs(x - self.mean[cls]) / std))

    def learn_one(self, x: np.ndarray, cls: int):
        if cls not in self.n:
            self.n[cls] = 0
            self.mean[cls] = np.zeros_like(x, dtype=float)
            self.M2[cls] = np.zeros_like(x, dtype=float)
            self._intra[cls], self._inter[cls] = [], []
        self.n[cls] += 1
        delta = x - self.mean[cls]
        self.mean[cls] += delta / self.n[cls]
        self.M2[cls] += delta * (x - self.mean[cls])
        self._intra[cls].append(self._dist(x, cls))
        for other_cls in self.mean:
            if other_cls != cls:
                self._inter[other_cls].append(self._dist(x, other_cls))

    def calibrate(self):
        WINDOW = 2000
        for cls in self.mean:
            intra = np.array(self._intra[cls][-WINDOW:])
            inter = np.array(self._inter.get(cls, [])[-WINDOW:])
            if len(intra) < 10 or len(inter) < 10:
                self.thresholds[cls] = (
                    float(np.percentile(intra, 95)) if len(intra) else 1.0
                )
                continue
            candidates = np.linspace(
                min(intra.min(), inter.min()), max(intra.max(), inter.max()), 400
            )
            best_youden, best_thr = -np.inf, candidates[-1]
            for t in candidates:
                youden = float(np.mean(intra <= t)) + float(np.mean(inter > t)) - 1.0
                if youden > best_youden:
                    best_youden, best_thr = youden, t
            self.thresholds[cls] = float(best_thr)

    def is_zero_day(self, x: np.ndarray, predicted_cls: int) -> bool:
        if predicted_cls not in self.mean:
            return True
        for cls in self.mean:
            if self._dist(x, cls) <= self.thresholds.get(cls, float("inf")):
                return False
        return True


def _entropy(proba_dict: dict) -> float:
    probs = np.array(list(proba_dict.values()))
    probs = probs[probs > 0]
    return float(-np.sum(probs * np.log2(probs + 1e-12)))


class EntropyOSR:
    def __init__(self, window_size: int = 2000):
        self._entropy_window: deque = deque(maxlen=window_size)
        self.threshold: float = 1.0

    def learn_one(self, proba_dict: dict):
        self._entropy_window.append(_entropy(proba_dict))
        if len(self._entropy_window) >= 10:
            self.threshold = float(np.percentile(list(self._entropy_window), 95))

    def calibrate(self):
        if len(self._entropy_window) >= 10:
            self.threshold = float(np.percentile(list(self._entropy_window), 95))

    def is_zero_day(self, proba_dict: dict) -> bool:
        return _entropy(proba_dict) > self.threshold


class PrequentialSelector:
    def __init__(
        self, window_size: int = 200, zero_day_label: int = 9, min_samples: int = 30
    ):
        self.ZERO_DAY, self.min_samples = zero_day_label, min_samples
        self.windows: dict = defaultdict(
            lambda: {d: deque(maxlen=window_size) for d in ("A", "B")}
        )

    def update(self, predicted_cls, y_true_mapped, pred_a, pred_b):
        w = self.windows[predicted_cls]
        w["A"].append((y_true_mapped, pred_a))
        w["B"].append((y_true_mapped, pred_b))

    def select(self, predicted_cls: int) -> str:
        w = self.windows[predicted_cls]
        if len(w["B"]) < self.min_samples:
            return "B"
        best_det, best_f1 = "B", -1.0
        for det, window in w.items():
            if len(window) < self.min_samples:
                continue
            y_t, y_p = [s[0] for s in window], [s[1] for s in window]
            try:
                f1 = f1_score(
                    y_t, y_p, labels=[self.ZERO_DAY], average="macro", zero_division=0
                )
            except:
                f1 = 0.0
            if f1 > best_f1:
                best_f1, best_det = f1, det
        return best_det


def run_experiment(
    zero_day_class: int,
    X_raw: pd.DataFrame,
    y_true: np.ndarray,
    numeric_cols: list,
    categ_cols: list,
    le_classes: np.ndarray,
    normal_class: int,
):
    zd_name = le_classes[zero_day_class]
    prefix = zd_name  # usado em todos os logs deste processo

    ZERO_DAY_LABEL, NORMAL_CLASS = 9, normal_class
    ZERO_DAY_CLASSES = [zero_day_class]
    KNOWN_CLASSES = [c for c in range(len(le_classes)) if c not in ZERO_DAY_CLASSES]

    labels_eval = KNOWN_CLASSES + [ZERO_DAY_LABEL]
    target_names = [f"class_{c}" for c in KNOWN_CLASSES] + ["zero_day"]

    log("Dividindo índices train/test/zero-day…", prefix)

    mask_known = ~np.isin(y_true, ZERO_DAY_CLASSES)
    idx_known, idx_zd = np.where(mask_known)[0], np.where(~mask_known)[0]
    split_point = int(len(idx_known) * 0.90)
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

    log("Pré-processando features (OrdinalEncoder)…", prefix)
    if categ_cols:
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
                    oe.transform(X_te_cat), columns=categ_cols, index=X_test_raw.index
                ),
            ],
            axis=1,
        )
        X_zd_proc = pd.concat(
            [
                X_zd_raw[numeric_cols],
                pd.DataFrame(
                    oe.transform(X_z_cat), columns=categ_cols, index=X_zd_raw.index
                ),
            ],
            axis=1,
        )
    else:
        X_train_proc, X_test_proc, X_zd_proc = (
            X_train_raw[numeric_cols],
            X_test_raw[numeric_cols],
            X_zd_raw[numeric_cols],
        )

    feature_names = X_train_proc.columns.tolist()
    X_train_arr = X_train_proc.values
    X_test_arr = X_test_proc.values
    X_zd_arr = X_zd_proc.values

    log("Inicializando HST, AdaBoost, OSR-A/B, Selector…", prefix)

    hst = compose.Pipeline(
        preprocessing.StandardScaler(),
        anomaly.HalfSpaceTrees(n_trees=10, height=11, window_size=100, seed=42),
    )

    clf = compose.Pipeline(
        preprocessing.StandardScaler(),
        ensemble.AdaBoostClassifier(
            model=tree.HoeffdingTreeClassifier(split_criterion="gini", max_depth=15),
            n_models=15,
            seed=42,
        ),
    )

    osr_a, osr_scaler, osr_b = (
        CentroidOSR(),
        StandardScaler(),
        EntropyOSR(window_size=2000),
    )

    selector = PrequentialSelector(
        window_size=200, zero_day_label=ZERO_DAY_LABEL, min_samples=30
    )

    auto_labeler = ConservativeAutoLabeler(
        confidence_threshold=0.90,
        history_window=50,
        min_history_count=3,
        zero_day_label=ZERO_DAY_LABEL,
    )

    n_normal = int((y_train == NORMAL_CLASS).sum())

    log(f"Warm-up HST com {n_normal} amostras normais…", prefix)

    for x_row in X_train_arr[y_train == NORMAL_CLASS]:
        hst.learn_one({feature_names[j]: x_row[j] for j in range(len(x_row))})

    log(f"Calculando scores HST no treino ({len(X_train_arr)} amostras)…", prefix)

    scores_train, labels_binary = [], []

    for i, x_row in enumerate(X_train_arr):
        xi = {feature_names[j]: x_row[j] for j in range(len(x_row))}
        scores_train.append(hst.score_one(xi))
        labels_binary.append(0 if y_train[i] == NORMAL_CLASS else 1)

    scores_arr = np.array(scores_train)
    thr_cands = np.linspace(scores_arr.min(), scores_arr.max(), 300)
    hst_threshold = float(
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
    log(f"Limiar HST calibrado: {hst_threshold:.6f}", prefix)

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

    log("Calibrando OSR-A (CentroidOSR)…", prefix)
    osr_a.calibrate()

    log("Calibrando OSR-B (EntropyOSR)…", prefix)
    osr_b.calibrate()

    X_full = pd.concat(
        [pd.DataFrame(X_test_arr), pd.DataFrame(X_zd_arr)], axis=0
    ).reset_index(drop=True)
    y_full_arr = np.concatenate([y_test, y_zd_sample])
    n_total = len(y_full_arr)

    log(f"Iniciando avaliação híbrida ({n_total} amostras)…", prefix)
    hybrid_true, hybrid_pred, total_ns, n_samples = [], [], 0, 0
    detector_usage = defaultdict(int)
    autolabel_counts = {"accepted": 0, "rejected": 0}
    expert_queue = deque(maxlen=100)

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
                if len(expert_queue) == 100:
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

    log("Treinando e avaliando baseline AdaBoost…", prefix)
    bl = compose.Pipeline(
        preprocessing.StandardScaler(),
        ensemble.AdaBoostClassifier(
            model=tree.HoeffdingTreeClassifier(split_criterion="gini", max_depth=15),
            n_models=15,
            seed=42,
        ),
    )
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
        fp = ZERO_DAY_LABEL if (max(probas.values()) if probas else 0) < 0.70 else pred
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


if __name__ == "__main__":
    NORMAL_CLASS = -1
    log("Iniciando experimento…")

    path = (
        "./ML_EdgeIIoT_SMOTE.csv"
        if os.path.exists("./ML_EdgeIIoT_SMOTE.csv")
        else "./ERENO-2.0-100K.csv"
    )
    log(f"Carregando dataset: {path}")

    df = pd.read_csv(path, low_memory=False)
    TARGET_COL = "Attack_type" if "Attack_type" in df.columns else "class"
    df = df.dropna(subset=[TARGET_COL])
    log(f"Dataset carregado: {len(df)} linhas, {len(df.columns)} colunas.")

    y_raw, X_raw = df[TARGET_COL].astype(str), df.drop(columns=[TARGET_COL])
    numeric_cols = X_raw.select_dtypes(include=[np.number]).columns.tolist()
    categ_cols = X_raw.select_dtypes(exclude=[np.number]).columns.tolist()

    constant_cols = [col for col in numeric_cols if X_raw[col].nunique() <= 1]
    if constant_cols:
        log(f"Removendo {len(constant_cols)} colunas constantes.")
        X_raw = X_raw.drop(columns=constant_cols)
        numeric_cols = [c for c in numeric_cols if c not in constant_cols]

    log("Aplicando log1p em colunas numéricas skewed…")
    for col in numeric_cols:
        if X_raw[col].min() >= 0 and X_raw[col].max() > 1000:
            X_raw[col] = np.log1p(X_raw[col])

    X_raw = X_raw.replace([np.inf, -np.inf], np.nan).fillna(0)
    X_raw[numeric_cols] = X_raw[numeric_cols].clip(lower=-1e6, upper=1e6)

    le = LabelEncoder()
    y_true = le.fit_transform(y_raw)
    for i, name in enumerate(le.classes_):
        if name.lower() == "normal":
            NORMAL_CLASS = i

    classes_to_test = [c for c in range(len(le.classes_)) if c != NORMAL_CLASS]
    log(
        f"Dataset: {path} | Normal class idx={NORMAL_CLASS} | "
        f"Classes a testar: {len(classes_to_test)}"
    )
    log(f"Classes: {list(le.classes_)}")

    results = []
    n_workers = max(1, os.cpu_count() - 2)
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
                NORMAL_CLASS,
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
        log("Nenhum resultado disponível")
