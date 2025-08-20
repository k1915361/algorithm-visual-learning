# profile_runner.py
import sys, cProfile, pstats, io
from manim import tempconfig
from _2_manim import LinkedListReverseScene

# --- ONE‑KEY TOGGLE ---
# Set MODE = 0 for low_quality  (like -pql)
# Set MODE = 1 for high_quality (1080p, 60fps)
MODE = int(sys.argv[1]) if len(sys.argv) > 1 else 0  # <- change 0 to 1 (one keystroke) to switch
QUALITY = ("low_quality", "high_quality")[MODE]

print(f"[profile_runner] Using quality preset: {QUALITY}")

with tempconfig({
    "quality": QUALITY,
    "disable_caching": True,  # keep True while profiling so file IO doesn't skew results
}):
    # Option A: quick run that writes stats to a file
    cProfile.run("LinkedListReverseScene().render()", "manim_profile.stats")

    # Option B: inline stats (same run again; comment out if you only want one render)
    pr = cProfile.Profile()
    pr.enable()
    LinkedListReverseScene().render()
    pr.disable()

    s = io.StringIO()
    p = pstats.Stats("manim_profile.stats").sort_stats("cumtime")
    p.print_stats(50)  # top 50 lines
    # print(s.getvalue())  # uncomment if you want to capture/inspect in code
