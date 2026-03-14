import time
import numpy as np
import pandas as pd
import concurrent.futures
import os
from collections import defaultdict, deque
import csv

from river import anomaly, compose, preprocessing, ensemble, tree
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder, StandardScaler
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import train_test_split


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
        self,
        predicted_cls: int,
        pred_a: int,
        pred_b: int,
        probas: dict,
    ) -> int | None:
        # Agora avalia apenas concordância entre A e B
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
        std = np.sqrt(self.M2[cls] / max(self.n[cls] - 1, 1))
        std = np.where(std < 1e-9, 1e-9, std)
        return float(np.mean(np.abs(x - self.mean[cls]) / std))

    def learn_one(self, x: np.ndarray, cls: int):
        if cls not in self.n:
            self.n[cls] = 0
            self.mean[cls] = np.zeros_like(x, dtype=float)
            self.M2[cls] = np.zeros_like(x, dtype=float)
            self._intra[cls] = []
            self._inter[cls] = []
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
                min(intra.min(), inter.min()),
                max(intra.max(), inter.max()),
                400,
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
        self,
        window_size: int = 200,
        zero_day_label: int = 9,
        min_samples: int = 30,
    ):
        self.window_size = window_size
        self.ZERO_DAY = zero_day_label
        self.min_samples = min_samples
        self.windows: dict = defaultdict(
            lambda: {d: deque(maxlen=window_size) for d in ("A", "B")}
        )

    def update(self, predicted_cls, y_true_mapped, pred_a, pred_b):
        w = self.windows[predicted_cls]
        w["A"].append((y_true_mapped, pred_a))
        w["B"].append((y_true_mapped, pred_b))

    def select(self, predicted_cls: int) -> str:
        w = self.windows[predicted_cls]
        # Se não houver dados suficientes, começa com B (Entropia) como padrão
        if len(w["B"]) < self.min_samples:
            return "B"

        best_detector, best_f1 = "B", -1.0
        for det, window in w.items():
            if len(window) < self.min_samples:
                continue
            y_t = [s[0] for s in window]
            y_p = [s[1] for s in window]
            try:
                f1 = f1_score(
                    y_t, y_p, labels=[self.ZERO_DAY], average="macro", zero_division=0
                )
            except Exception:
                f1 = 0.0
            if f1 > best_f1:
                best_f1, best_detector = f1, det
        return best_detector


def run_experiment(
    zero_day_class: int,
    X_proc: pd.DataFrame,
    y_true: np.ndarray,
    feature_names: list,
    le_classes: np.ndarray,
    normal_class: int,
):
    zd_name = le_classes[zero_day_class]
    ZERO_DAY_LABEL = 9
    ZERO_DAY_CLASSES = [zero_day_class]
    KNOWN_CLASSES = [c for c in range(len(le_classes)) if c not in ZERO_DAY_CLASSES]
    NORMAL_CLASS = normal_class

    mask_known = ~np.isin(y_true, ZERO_DAY_CLASSES)
    X_zd = X_proc[np.isin(y_true, ZERO_DAY_CLASSES)]
    y_zd = y_true[np.isin(y_true, ZERO_DAY_CLASSES)]

    X_train, X_test, y_train, y_test = train_test_split(
        X_proc[mask_known],
        y_true[mask_known],
        test_size=0.1,
        random_state=42,
        stratify=y_true[mask_known],
    )
    _, counts = np.unique(y_test, return_counts=True)
    X_zd_sample = X_zd.iloc[: max(counts)]
    y_zd_sample = y_zd[: max(counts)]

    X_train_arr = X_train.reset_index(drop=True).values
    X_test_arr = X_test.reset_index(drop=True).values

    hst = compose.Pipeline(
        preprocessing.MinMaxScaler(),
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
    osr_a = CentroidOSR()
    osr_scaler = StandardScaler()
    osr_b = EntropyOSR(window_size=2000)

    selector = PrequentialSelector(
        window_size=200, zero_day_label=ZERO_DAY_LABEL, min_samples=30
    )
    auto_labeler = ConservativeAutoLabeler(
        confidence_threshold=0.90,
        history_window=50,
        min_history_count=3,
        zero_day_label=ZERO_DAY_LABEL,
    )

    for x_row in X_train_arr[y_train == NORMAL_CLASS]:
        xi = {feature_names[j]: x_row[j] for j in range(len(x_row))}
        hst.learn_one(xi)

    scores_train, labels_binary = [], []
    for i, x_row in enumerate(X_train_arr):
        xi = {feature_names[j]: x_row[j] for j in range(len(x_row))}
        scores_train.append(hst.score_one(xi))
        labels_binary.append(0 if y_train[i] == NORMAL_CLASS else 1)

    scores_arr = np.array(scores_train)
    labels_arr = np.array(labels_binary)
    thr_cands = np.linspace(scores_arr.min(), scores_arr.max(), 300)
    f1s = [
        f1_score(labels_arr, (scores_arr > t).astype(int), zero_division=0)
        for t in thr_cands
    ]
    hst_threshold = float(thr_cands[np.argmax(f1s)])

    osr_scaler.fit(X_train_arr[y_train != NORMAL_CLASS])

    for i, x_row in enumerate(X_train_arr):
        xi = {feature_names[j]: x_row[j] for j in range(len(x_row))}
        cls = y_train[i]
        clf.learn_one(xi, cls)
        if cls != NORMAL_CLASS:
            x_sc = osr_scaler.transform(x_row.reshape(1, -1))[0]
            osr_a.learn_one(x_sc, cls)
            probas = clf.predict_proba_one(xi)
            if probas:
                osr_b.learn_one(probas)

    osr_a.calibrate()
    osr_b.calibrate()

    X_full = pd.concat(
        [pd.DataFrame(X_test_arr, columns=X_proc.columns), X_zd_sample], axis=0
    ).reset_index(drop=True)
    y_full_arr = np.concatenate([y_test, y_zd_sample])

    hybrid_true, hybrid_pred = [], []
    total_ns = 0
    n_samples = 0
    detector_usage = defaultdict(int)
    autolabel_counts = {"accepted": 0, "rejected": 0}

    RECALIB_A_INTERVAL = 100
    samples_since_recalib_a = 0

    for i, x_row in enumerate(X_full.values):
        t0 = time.time_ns()
        xi = {feature_names[j]: x_row[j] for j in range(len(x_row))}
        y_full = y_full_arr[i]
        y_bin = 0 if y_full == NORMAL_CLASS else 1

        hst_score = hst.score_one(xi)
        predicted_bin = 1 if hst_score > hst_threshold else 0

        if predicted_bin == 1 and y_bin == 1:
            predicted_cls = clf.predict_one(xi)
            probas = clf.predict_proba_one(xi) or {}
            x_sc = osr_scaler.transform(x_row.reshape(1, -1))[0]

            if predicted_cls == NORMAL_CLASS:
                final_pred = ZERO_DAY_LABEL
                pred_a = pred_b = ZERO_DAY_LABEL
                total_ns += time.time_ns() - t0
            else:
                det = selector.select(predicted_cls)
                detector_usage[det] += 1

                # Executa o detector selecionado para a predição final
                if det == "A":
                    pred_a = (
                        ZERO_DAY_LABEL
                        if osr_a.is_zero_day(x_sc, predicted_cls)
                        else predicted_cls
                    )
                    final_pred = pred_a
                else:
                    pred_b = (
                        ZERO_DAY_LABEL
                        if (probas and osr_b.is_zero_day(probas))
                        else predicted_cls
                    )
                    final_pred = pred_b

                total_ns += time.time_ns() - t0

                # Calcula o outro para fins de Auto-Rotulação e Selector
                if det != "A":
                    pred_a = (
                        ZERO_DAY_LABEL
                        if osr_a.is_zero_day(x_sc, predicted_cls)
                        else predicted_cls
                    )
                if det != "B":
                    pred_b = (
                        ZERO_DAY_LABEL
                        if (probas and osr_b.is_zero_day(probas))
                        else predicted_cls
                    )

                auto_cls = auto_labeler.evaluate(
                    predicted_cls=predicted_cls,
                    pred_a=pred_a,
                    pred_b=pred_b,
                    probas=probas,
                )

                if auto_cls is not None and auto_cls != NORMAL_CLASS:
                    autolabel_counts["accepted"] += 1
                    osr_a.learn_one(x_sc, auto_cls)
                    samples_since_recalib_a += 1
                    if samples_since_recalib_a >= RECALIB_A_INTERVAL:
                        osr_a.calibrate()
                        samples_since_recalib_a = 0

                    if probas:
                        osr_b.learn_one(probas)

                    clf.learn_one(xi, auto_cls)
                    hst.learn_one(xi)
                else:
                    autolabel_counts["rejected"] += 1

            y_true_mapped = ZERO_DAY_LABEL if y_full in ZERO_DAY_CLASSES else y_full
            hybrid_true.append(y_true_mapped)
            hybrid_pred.append(final_pred)

            if predicted_cls != NORMAL_CLASS:
                selector.update(predicted_cls, y_true_mapped, pred_a, pred_b)
        else:
            total_ns += time.time_ns() - t0

        n_samples += 1

    avg_latency_us = (total_ns / max(n_samples, 1)) / 1_000
    labels_eval = KNOWN_CLASSES + [ZERO_DAY_LABEL]
    target_names = [f"class_{c}" for c in KNOWN_CLASSES] + ["zero_day"]

    # Métricas Híbrido
    f1_hybrid_zd = 0.0
    f1_hybrid_known = 0.0
    f1_hybrid_macro = 0.0

    if hybrid_true:
        report = classification_report(
            hybrid_true,
            hybrid_pred,
            labels=labels_eval,
            target_names=target_names,
            zero_division=0,
            output_dict=True,
        )
        f1_hybrid_zd = report["zero_day"]["f1-score"]
        known_f1s = [
            report[tgt]["f1-score"] for tgt in target_names if tgt != "zero_day"
        ]
        f1_hybrid_known = np.mean(known_f1s)
        f1_hybrid_macro = report["macro avg"]["f1-score"]

    # Baseline (Idêntico ao original)
    bl = compose.Pipeline(
        preprocessing.StandardScaler(),
        ensemble.AdaBoostClassifier(
            model=tree.HoeffdingTreeClassifier(split_criterion="gini", max_depth=15),
            n_models=15,
            seed=42,
        ),
    )
    for i, x_row in enumerate(X_train_arr):
        xi = {feature_names[j]: x_row[j] for j in range(len(x_row))}
        bl.learn_one(xi, y_train[i])

    bl_true, bl_pred, bl_ns, bl_n = [], [], 0, 0
    for i, x_row in enumerate(X_full.values):
        t0 = time.time_ns()
        xi = {feature_names[j]: x_row[j] for j in range(len(x_row))}
        pred = bl.predict_one(xi)
        probas = bl.predict_proba_one(xi)
        mc = max(probas.values()) if probas else 0
        fp = ZERO_DAY_LABEL if mc < 0.70 else pred
        bl_ns += time.time_ns() - t0
        bl_n += 1
        yf = y_full_arr[i]
        bl_true.append(ZERO_DAY_LABEL if yf in ZERO_DAY_CLASSES else yf)
        bl_pred.append(fp)

    bl_latency_us = (bl_ns / max(bl_n, 1)) / 1_000
    report_bl = classification_report(
        bl_true,
        bl_pred,
        labels=labels_eval,
        target_names=target_names,
        zero_division=0,
        output_dict=True,
    )
    f1_bl_zd = report_bl["zero_day"]["f1-score"]
    known_f1s_bl = [
        report_bl[tgt]["f1-score"] for tgt in target_names if tgt != "zero_day"
    ]
    f1_bl_known = np.mean(known_f1s_bl)
    f1_bl_macro = report_bl["macro avg"]["f1-score"]

    accept_rate = autolabel_counts["accepted"] / max(
        autolabel_counts["accepted"] + autolabel_counts["rejected"], 1
    )
    det_str = " ".join(f"{k}:{v}" for k, v in sorted(detector_usage.items()))

    print(
        f"[ZD:{zd_name:<22}] Híb ZD:{f1_hybrid_zd:.4f} (Knw:{f1_hybrid_known:.4f}) | Bas ZD:{f1_bl_zd:.4f} (Knw:{f1_bl_known:.4f}) | Δ_ZD:{f1_hybrid_zd - f1_bl_zd:+.4f} | AL:{accept_rate:.1%} | det:[{det_str}]"
    )

    return {
        "Zero_Day_Class": zd_name,
        "Hybrid_F1_ZD": f1_hybrid_zd,
        "Baseline_F1_ZD": f1_bl_zd,
        "Delta_F1_ZD": f1_hybrid_zd - f1_bl_zd,
        "Hybrid_F1_Known": f1_hybrid_known,
        "Baseline_F1_Known": f1_bl_known,
        "Hybrid_F1_Macro": f1_hybrid_macro,
        "Baseline_F1_Macro": f1_bl_macro,
        "Hybrid_Latency_us": avg_latency_us,
        "Baseline_Latency_us": bl_latency_us,
        "AutoLabel_AcceptRate": accept_rate,
    }


if __name__ == "__main__":
    NORMAL_CLASS = -1
    print("Iniciando Pipeline Híbrido\n")

    with open(
        "../../ML-EdgeIIoT-dataset-CLEANED.csv", "r", encoding="utf-8", errors="replace"
    ) as f:
        reader = csv.reader(f)
        header = next(reader)
        dados = [row for row in reader]

    df = pd.DataFrame(dados, columns=header)
    TARGET_COL = "Attack_type"

    # for col in df.columns:
    #     if col != TARGET_COL:
    #         df[col] = pd.to_numeric(df[col], errors="coerce")

    df = (
        df.groupby(TARGET_COL)
        .apply(lambda x: x.sample(min(len(x), 20_000), random_state=42))
        .reset_index(drop=True)
    )
    if "Time" in df.columns:
        df = df.sort_values(by="Time").reset_index(drop=True)

    y_raw = df[TARGET_COL].astype(str)
    X_raw = df.drop(columns=[TARGET_COL])

    le = LabelEncoder()
    y_true = le.fit_transform(y_raw)
    for i, cls_name in enumerate(le.classes_):
        if cls_name.lower() == "normal":
            NORMAL_CLASS = i

    numeric_cols = X_raw.select_dtypes(include=[np.number]).columns.tolist()
    categ_cols = X_raw.select_dtypes(exclude=[np.number]).columns.tolist()

    oe = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    X_cat_enc = pd.DataFrame(
        oe.fit_transform(X_raw[categ_cols].fillna("##MISSING##")),
        columns=categ_cols,
        index=X_raw.index,
    )
    X_proc = pd.concat([X_raw[numeric_cols], X_cat_enc], axis=1)
    feature_names = X_proc.columns.tolist()

    classes_to_test = [c for c in range(len(le.classes_)) if c != NORMAL_CLASS]
    results = []

    with concurrent.futures.ProcessPoolExecutor(max_workers=10) as executor:
        futures = {
            executor.submit(
                run_experiment,
                zd,
                X_proc,
                y_true,
                feature_names,
                le.classes_,
                NORMAL_CLASS,
            ): zd
            for zd in classes_to_test
        }
        for future in concurrent.futures.as_completed(futures):
            try:
                results.append(future.result())
            except Exception as exc:
                print(f"Erro: {exc}")

    SEP = "=" * 85
    print(f"\n{SEP}\nRESULTADO — Pipeline Híbrido\n{SEP}")

    df_r = pd.DataFrame(results).sort_values("Zero_Day_Class")
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 1000)

    print(df_r.to_string(index=False))
    print(SEP)
    print(f"Média F1 ZD Híbrido   : {df_r['Hybrid_F1_ZD'].mean():.4f}")
    print(f"Média F1 ZD Baseline  : {df_r['Baseline_F1_ZD'].mean():.4f}")
    print(f"Delta médio ZD        : {df_r['Delta_F1_ZD'].mean():+.4f}")
    print(f"Latência híbrido      : {df_r['Hybrid_Latency_us'].mean():.1f} µs/sample")
    print(f"Taxa de auto-rotulação: {df_r['AutoLabel_AcceptRate'].mean():.1%}")
    print(SEP)
