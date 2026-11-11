"""Small logging helpers."""
import time


def log(*a):
    print(time.strftime("[%H:%M:%S]"), *a, flush=True)


def fmt_time(s):
    s = int(max(0, s))
    return f"{s // 3600}:{s % 3600 // 60:02d}:{s % 60:02d}"
