import numpy as np


class CentroidOSR:
    def __init__(self, window: int = 2000, n_candidates: int = 400):
        self._window = window
        self._n_candidates = n_candidates
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
        for cls in self.mean:
            intra = np.array(self._intra[cls][-self._window :])
            inter = np.array(self._inter.get(cls, [])[-self._window :])
            if len(intra) < 10 or len(inter) < 10:
                self.thresholds[cls] = (
                    float(np.percentile(intra, 95)) if len(intra) else 1.0
                )
                continue
            candidates = np.linspace(
                min(intra.min(), inter.min()),
                max(intra.max(), inter.max()),
                self._n_candidates,
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
