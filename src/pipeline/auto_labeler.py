from collections import deque


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
