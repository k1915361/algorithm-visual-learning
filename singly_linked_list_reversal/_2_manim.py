from manim import *

# Dev-time speed knobs
from manim import config
config.disable_caching = True             # hashing/json.dumps dominated your profile

# CSV timing helper (you already created this in profiling_helpers.py)
from profiling_helpers import tplay_csv

# -----------------------------
# Captions (PREBUILT once)
# -----------------------------
CAP = {
    1: MarkupText("<span>1.</span> Save <span color='#4ea3ff'>next</span> = curr.next").scale(0.6).to_edge(DOWN),
    2: MarkupText("<span>2.</span> Reverse link: curr.next = prev").scale(0.6).to_edge(DOWN),
    3: MarkupText("<span>3.</span> Move <span color='#ffd54a'>prev</span> = curr").scale(0.6).to_edge(DOWN),
    4: MarkupText("<span>4.</span> Move <span color='#6de97c'>curr</span> = next").scale(0.6).to_edge(DOWN),
}
CAP_DONE = MarkupText("<span>Done:</span> New head is <span color='#ffd54a'>prev</span>.").scale(0.6).to_edge(DOWN)

def flash(mobj, color=ORANGE, scale_factor=1.02, run_time=0.4):
    return Indicate(
        mobj,
        color=color,
        rate_func=there_and_back,
        scale_factor=scale_factor,
        run_time=run_time,
    )

import numpy as np
from manim import ValueTracker, smooth

def glide_retarget(scene: Scene, arr: Arrow, src_obj, dst_obj,
                   src_dir=LEFT, dst_dir=RIGHT, run_time=0.6, rate_func=smooth):
    # Current endpoints (where the arrow is now)
    s0 = arr.get_start()
    e0 = arr.get_end()

    # Target endpoints (where you want it to go)
    s1 = src_obj.get_edge_center(src_dir)
    e1 = dst_obj.get_edge_center(dst_dir)

    t = ValueTracker(0.0)

    def lerp(a, b, u):
        return a * (1 - u) + b * u

    def updater(m: Arrow):
        u = t.get_value()
        s = lerp(s0, s1, u)
        e = lerp(e0, e1, u)
        m.put_start_and_end_on(s, e)

    arr.add_updater(updater)
    tplay_csv(scene, t.animate.set_value(1.0), run_time=run_time, rate_func=rate_func)
    arr.remove_updater(updater)


# -----------------------------
# Lightweight node + pointer views
# -----------------------------
class NodeView(VGroup):
    def __init__(self, value, **kwargs):
        super().__init__(**kwargs)
        box = RoundedRectangle(corner_radius=0.15, height=0.7, width=1.0).set_stroke(WHITE, 2)
        label = Text(str(value)).scale(0.8).move_to(box.get_center())
        self.box, self.label = box, label
        self.add(box, label)

class Pointer(VGroup):
    """Pointer with small label-cache to avoid recreating Text repeatedly."""
    def __init__(self, name, color, y_offset=0.0):
        super().__init__()
        self.name, self.color, self.y_offset = name, color, y_offset
        self.tag = Text(name).scale(0.7).set_color(color)
        self.dot = Dot(color=color)
        self._label_cache = {name: self.tag.copy()}
        self.add(self.tag, self.dot)

    def _label_text(self, label_text):
        return label_text if label_text is not None else self.name

    def _get_or_make_label(self, label_text: str):
        if label_text not in self._label_cache:
            self._label_cache[label_text] = Text(label_text).scale(0.7).set_color(self.color)
        return self._label_cache[label_text]

    def place_instant(self, ref_mobj, label_text=None, base_buff=0.45):
        txt = self._get_or_make_label(self._label_text(label_text))
        self.tag.become(txt)
        self.tag.next_to(ref_mobj, DOWN, buff=base_buff)
        self.tag.shift(DOWN * self.y_offset)
        self.dot.move_to(self.tag.get_bottom() + DOWN * 0.12)

    def move_under_animations(self, ref_mobj, label_text=None, base_buff=0.45):
        target = self._get_or_make_label(self._label_text(label_text)).copy()
        target.next_to(ref_mobj, DOWN, buff=base_buff)
        target.shift(DOWN * self.y_offset)
        return [
            Transform(self.tag, target),
            self.dot.animate.move_to(target.get_bottom() + DOWN * 0.12),
        ]

    def pulse(self, scale_factor=1.2, run_time=0.2):
        return AnimationGroup(
            ScaleInPlace(self.tag, scale_factor, rate_func=there_and_back, run_time=run_time * 2),
            ScaleInPlace(self.dot, scale_factor, rate_func=there_and_back, run_time=run_time * 2),
            lag_ratio=0.0,
        )

# -----------------------------
# Scene
# -----------------------------
class LinkedListReverseScene(Scene):
    def construct(self):
        # layout knobs
        values = [1, 2, 3]
        gap = 2.0
        y_nodes = 1.5
        pause = 0.6

        # caption host
        caption = MarkupText("").scale(0.6).to_edge(DOWN)
        self.add(caption)

        # nodes
        nodes = VGroup(*[NodeView(v) for v in values]).arrange(RIGHT, buff=gap, aligned_edge=UP)
        nodes.shift(UP * y_nodes)
        tplay_csv(self, *[FadeIn(n, shift=DOWN, lag_ratio=0.05) for n in nodes], run_time=1.0)

        # right-side None marker
        none_marker = Text("None").scale(0.7)
        none_marker.next_to(nodes[-1], RIGHT, buff=gap * 0.6).align_to(nodes[-1], UP)
        none_box = SurroundingRectangle(none_marker, buff=0.08).set_opacity(0).set_stroke(width=0)
        self.add(none_box, none_marker)

        # layering
        for n in nodes:
            n.set_z_index(10)
        none_marker.set_z_index(5)

        # helpers
        def arrow_from_to(src_obj, dst_obj, *, src_dir=RIGHT, dst_dir=LEFT, pad=0.10, path_arc=0.0, z_index=None):
            # arrow turns orange
            start = src_obj.get_edge_center(src_dir)
            end = dst_obj.get_edge_center(dst_dir)
            arr = Arrow(start, end, buff=pad, stroke_width=6, max_tip_length_to_length_ratio=0.2, path_arc=path_arc)
            if z_index is not None:
                arr.set_z_index(z_index)
            return arr

        def set_caption_i(i: int, t=pause):
            caption.become(CAP[i])
            self.wait(t)

        def highlight(i, color=GREEN):
            return nodes[i].box.animate.set_stroke(color, 4)

        def target_of(idx):
            return nodes[idx] if idx is not None else none_marker

        def label_of(name, idx):
            return name if idx is not None else f"{name}=None"

        def move_and_pulse(ptr, target, label):
            # one play: move + pulse (list-merge)
            tplay_csv(self, *ptr.move_under_animations(target, label), run_time=0.6)

            # pulse AFTER moving
            tplay_csv(self, ptr.pulse(), run_time=0.25)

        def blink_line(i):
            return flash(code[i], color=YELLOW)

        def retarget(arr, src_obj, dst_obj, src_dir=LEFT, dst_dir=RIGHT):
            s = src_obj.get_edge_center(src_dir)
            e = dst_obj.get_edge_center(dst_dir)
            arr.put_start_and_end_on(s, e)
            return arr

        # initial "next" arrows
        next_arrows = []
        for i in range(len(nodes)):
            a = arrow_from_to(nodes[i].box, nodes[i + 1].box) if i < len(nodes) - 1 else arrow_from_to(nodes[i].box, none_box)
            a.set_z_index(0)
            next_arrows.append(a)
        tplay_csv(self, *[Create(a) for a in next_arrows], run_time=1.0)

        # pointers
        prev_ptr = Pointer("prev", YELLOW, y_offset=0.0)
        curr_ptr = Pointer("curr", GREEN, y_offset=0.5)
        next_ptr = Pointer("next", BLUE, y_offset=1.0)

        prev_ptr.place_instant(none_marker, "prev=None")
        curr_ptr.place_instant(nodes[0], "curr")
        next_ptr.place_instant(nodes[1] if len(nodes) > 1 else none_marker, "next" if len(nodes) > 1 else "next=None")

        tplay_csv(self, FadeIn(prev_ptr), FadeIn(curr_ptr), FadeIn(next_ptr), run_time=0.6)

        # code block
        code = Paragraph(
            "next = curr.next",
            "curr.next = prev",
            "prev, curr = curr, next",
            alignment="left",
            line_spacing=0.55,
        ).scale(0.45).to_corner(DL)
        code.set_opacity(0.95)
        bg = BackgroundRectangle(code, buff=0.15, corner_radius=0.12, fill_opacity=0.15)
        self.add(bg, code)

        # algorithm state
        prev_idx, curr_idx = None, 0

        def flip_arrow(i, new_prev_idx):
            old = next_arrows[i]
            # retarget existing arrow (avoid ReplacementTransform)
            # would it be better to separate this into if else statement?
            # I need to give smooth animation rather than sudden flips of arrows 
            
            if new_prev_idx is not None:
                dst_node = nodes[new_prev_idx].box
                dst_dir = RIGHT
            else:
                dst_node = none_box
                dst_dir = LEFT
                
            glide_retarget(self, old, nodes[i].box, dst_node, src_dir=LEFT, dst_dir=dst_dir, run_time=0.6)

            tplay_csv(self, flash(old))

            return old  # keep list entry pointing to the same arrow

        # iterate
        while curr_idx is not None:
            next_idx = curr_idx + 1 if curr_idx < len(values) - 1 else None

            set_caption_i(1)
            tplay_csv(
                self,
                nodes[curr_idx].box.animate.set_stroke(GREEN, 4),
                run_time=0.6,
                rate_func=there_and_back,   # pass as kwarg (avoid builder.set_rate_func)
            )

            # move next pointer
            move_and_pulse(next_ptr, target_of(next_idx), label_of("next", next_idx))
            tplay_csv(self, blink_line(0))

            set_caption_i(2)
            next_arrows[curr_idx] = flip_arrow(curr_idx, prev_idx)

            set_caption_i(3)
            move_and_pulse(prev_ptr, nodes[curr_idx], "prev")
            tplay_csv(self, blink_line(1))

            set_caption_i(4)
            move_and_pulse(curr_ptr, target_of(next_idx), label_of("curr", next_idx))
            tplay_csv(self, blink_line(2))

            prev_idx, curr_idx = curr_idx, next_idx
            self.wait(pause * 0.6)
            if curr_idx is None:
                break

        # done caption
        caption.become(CAP_DONE)
        self.wait(1.2)

        # payoff
        brace = Brace(nodes, direction=UP, buff=0.3)
        final_text = Text("reversed").scale(0.6).next_to(brace, UP, buff=0.2)
        tplay_csv(self, GrowFromCenter(brace), FadeIn(final_text, shift=UP * 0.2), run_time=0.8)
        self.wait(2.3)

        self.wait(0.8)
        tplay_csv(self, caption.animate.set_opacity(0), run_time=0.2)
        tplay_csv(self, FadeOut(caption, run_time=0.2))

        payoff = MarkupText(
            "while curr: next=curr.next; curr.next=prev; prev=curr; curr=next",
            font="Consolas",
        ).scale(0.45).to_edge(DOWN)
        tplay_csv(self, FadeOut(code), FadeOut(bg), FadeIn(payoff, shift=UP), run_time=0.35)

        # complexity caption
        complexity = Text("O(1) space · O(n) time").scale(0.4).next_to(caption, DOWN, buff=0.12)
        tplay_csv(self, FadeIn(complexity, shift=UP * 0.2), run_time=0.25)
        self.wait(5.0)
