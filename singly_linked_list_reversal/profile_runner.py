# profile_runner.py
import cProfile, pstats, io
from manim import tempconfig
from _2_manim import LinkedListReverseScene  

with tempconfig({"quality": "low_quality", "disable_caching": True}):  # same as -pql
    cProfile.run("LinkedListReverseScene().render()", "manim_profile.stats")
    pr = cProfile.Profile()
    pr.enable()
    LinkedListReverseScene().render()
    pr.disable()

    s = io.StringIO()
    p = pstats.Stats("manim_profile.stats").sort_stats("cumtime")

    p.print_stats(50)  # top 50 lines
    # print(s.getvalue())
