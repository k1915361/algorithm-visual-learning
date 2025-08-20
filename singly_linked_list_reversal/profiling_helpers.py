import csv, os, time, inspect, statistics as stats
from typing import Iterable
from manim import Scene

CSV_PATH = "manim_timing.csv"

def _callsite():
    f = inspect.currentframe().f_back.f_back  # skip tplay_csv + wrapper
    info = inspect.getframeinfo(f)
    return info.filename, info.lineno, (info.code_context[0].strip() if info.code_context else "")

def tplay_csv(scene: Scene, *anims, **kw):
    """Scene.play wrapper that logs timing to CSV with file/line callsite."""
    fn, ln, ctx = _callsite()
    t0 = time.perf_counter()
    scene.play(*anims, **kw)
    dt = time.perf_counter() - t0

    newfile = not os.path.exists(CSV_PATH)
    with open(CSV_PATH, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if newfile:
            w.writerow(["file", "line", "seconds", "kwargs", "context"])
        # keep kwargs short-ish in the file
        w.writerow([fn, ln, f"{dt:.6f}", repr(kw) if kw else "", ctx])
    print(f"[tplay_csv] {os.path.basename(fn)}:{ln} {dt:.3f}s")
    return dt

def summarize_csv(path: str = CSV_PATH, by=("file", "line")):
    """Print a table of count/avg/p50/p90/max grouped by callsite."""
    if not os.path.exists(path):
        print("No CSV found.")
        return
    rows = []
    import csv
    with open(path, newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            try:
                rows.append(row)
            except Exception:
                pass
    if not rows:
        print("No rows.")
        return

    # group
    from collections import defaultdict
    grp = defaultdict(list)
    for row in rows:
        key = tuple(row[k] for k in by)
        grp[key].append(float(row["seconds"]))

    # print
    print(f"\nSummary by {by}:")
    print(f"{'site':50}  {'n':>4}  {'avg':>7}  {'p50':>7}  {'p90':>7}  {'max':>7}")
    for key, vals in sorted(grp.items(), key=lambda kv: -sum(kv[1])):
        n = len(vals)
        avg = sum(vals)/n
        p50 = stats.quantiles(vals, n=2)[0] if n > 1 else vals[0]
        p90 = stats.quantiles(vals, n=10)[8] if n >= 10 else max(vals)
        mx = max(vals)
        site = f"{os.path.basename(key[0])}:{key[1]}"
        print(f"{site:50}  {n:4d}  {avg:7.3f}  {p50:7.3f}  {p90:7.3f}  {mx:7.3f}")
