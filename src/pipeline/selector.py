from collections import defaultdict, deque

from sklearn.metrics import f1_score


class PrequentialSelector:
    def __init__(
        self,
        window_size: int = 200,
        zero_day_label: int = 9,
        min_samples: int = 30,
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
                    y_t,
                    y_p,
                    labels=[self.ZERO_DAY],
                    average="macro",
                    zero_division=0,
                )
            except Exception:
                f1 = 0.0
            if f1 > best_f1:
                best_f1, best_det = f1, det
        return best_det
