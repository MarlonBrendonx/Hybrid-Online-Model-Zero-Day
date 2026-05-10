import time


def log(msg: str, prefix: str = "") -> None:
    ts = time.strftime("%H:%M:%S")
    tag = f"[{prefix}] " if prefix else ""
    print(f"[{ts}] {tag}{msg}", flush=True)
