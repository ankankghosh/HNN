"""Shared statistics helpers: mean +/- standard error, paired differences, win counts."""
import numpy as np


def mean_se(v):
    v = np.asarray(v, float)
    return float(v.mean()), float(v.std(ddof=1) / np.sqrt(len(v))) if len(v) > 1 else float("nan")


def fmt(v, d=2):
    m, s = mean_se(v)
    return f"{m:.{d}f} +/- {s:.{d}f}"


def paired(a, b):
    """a, b: dict seed -> value. Returns dict with mean diff (a-b), s.e., t statistic, wins of a, n."""
    seeds = sorted(set(a) & set(b))
    d = np.array([a[s] - b[s] for s in seeds], float)
    m, se = mean_se(d)
    return dict(n=len(seeds), diff=m, se=se, t=m / se if se and se > 0 else float("nan"),
                wins=int((d > 0).sum()), ties=int((d == 0).sum()), losses=int((d < 0).sum()))


def fmt_paired(a, b, d=2):
    p = paired(a, b)
    return f"{p['diff']:+.{d}f} +/- {p['se']:.{d}f} (t={p['t']:.1f}, W/T/L {p['wins']}/{p['ties']}/{p['losses']}, n={p['n']})"
