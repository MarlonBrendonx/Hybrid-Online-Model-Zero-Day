import numpy as np
from collections import deque


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
