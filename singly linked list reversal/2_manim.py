from manim import *

class NodeView(VGroup):
    def __init__(self, value, **kwargs):
        super().__init__(**kwargs)
        box = RoundedRectangle(corner_radius=0.15, height=0.7, width=1.0).set_stroke(WHITE, 2)
        label = Text(str(value)).scale(0.8).move_to(box.get_center())
        self.box, self.label = box, label
        self.add(box, label)

class Pointer(VGroup):
    def __init__(self, name, color, y_offset=0.0):
        super().__init__()
        self.name, self.color, self.y_offset = name, color, y_offset
        self.tag = Text(name).scale(0.7).set_color(color)
        self.dot = Dot(color=color)
        self.add(self.tag, self.dot)

    def _label_text(self, label_text):
        return label_text if label_text is not None else self.name

    def place_instant(self, ref_mobj, label_text=None, base_buff=0.45):
        txt = Text(self._label_text(label_text)).scale(0.7).set_color(self.color)
        self.tag.become(txt)
        self.tag.next_to(ref_mobj, DOWN, buff=base_buff)
        self.tag.shift(DOWN * self.y_offset)
        self.dot.move_to(self.tag.get_bottom() + DOWN * 0.12)

    def move_under_animations(self, ref_mobj, label_text=None, base_buff=0.45):
        target = Text(self._label_text(label_text)).scale(0.7).set_color(self.color)
        target.next_to(ref_mobj, DOWN, buff=base_buff)
        target.shift(DOWN * self.y_offset)
        return [
            Transform(self.tag, target),
            self.dot.animate.move_to(target.get_bottom() + DOWN * 0.12),
        ]

    def pulse(self, scale_factor=1.2, run_time=0.2):
        return AnimationGroup(
            ScaleInPlace(self.tag, scale_factor, rate_func=there_and_back, run_time=run_time*2),
            ScaleInPlace(self.dot, scale_factor, rate_func=there_and_back, run_time=run_time*2),
            lag_ratio=0.0,
        )


class LinkedListReverseScene(Scene):
    def construct(self):
        # --- layout knobs ---
        values = [1, 2, 3]
        gap = 2.0
        y_nodes = 1.5
        pause = 0.6
        fast = 0.5

        # --- nodes ---
        nodes = VGroup(*[NodeView(v) for v in values]).arrange(RIGHT, buff=gap, aligned_edge=UP)
        nodes.shift(UP * y_nodes)
        self.play(*[FadeIn(n, shift=DOWN, lag_ratio=0.05) for n in nodes], run_time=1.0)

        # right-side None marker
        none_marker = Text("None").scale(0.7)
        none_marker.next_to(nodes[-1], RIGHT, buff=gap * 0.6).align_to(nodes[-1], UP)
        none_box = SurroundingRectangle(none_marker, buff=0.08).set_opacity(0).set_stroke(width=0)
        self.add(none_box, none_marker)

        # layering
        for n in nodes: n.set_z_index(10)
        none_marker.set_z_index(5)

        # helpers
        def arrow_from_to(src_obj, dst_obj, *, src_dir=RIGHT, dst_dir=LEFT, pad=0.10, path_arc=0.0, z_index=None):
            start = src_obj.get_edge_center(src_dir)
            end   = dst_obj.get_edge_center(dst_dir)
            arr = Arrow(start, end, buff=pad, stroke_width=6, max_tip_length_to_length_ratio=0.2, path_arc=path_arc)
            if z_index is not None: arr.set_z_index(z_index)
            return arr

        def set_caption(html, t=pause):
            nonlocal caption
            caption.become(MarkupText(html).scale(0.6).to_edge(DOWN))
            self.wait(t)

        def highlight(i, color=GREEN):
            return nodes[i].box.animate.set_stroke(color, 4)

        def target_of(idx): 
            return nodes[idx] if idx is not None else none_marker

        def label_of(name, idx): 
            return name if idx is not None else f"{name}=None"

        def move_and_pulse(ptr, target, label):
            self.play(*ptr.move_under_animations(target, label))
            self.play(ptr.pulse())

        def blink_line(i):
            """
            # i: index like 0,1,2
            """
            lines = code.submobjects if hasattr(code, "submobjects") else [code]  # fallback
            return code.animate.set_color_by_t2c({   # lightweight per-line tint via substr
                "next = curr.next": YELLOW if i==0 else WHITE,
                "curr.next = prev": YELLOW if i==1 else WHITE,
                "prev, curr = curr, next": YELLOW if i==2 else WHITE,
            })

        # initial next arrows
        next_arrows = []
        for i in range(len(nodes)):
            if i < len(nodes) - 1:
                a = arrow_from_to(nodes[i].box, nodes[i+1].box)
            else:
                a = arrow_from_to(nodes[i].box, none_box)
            a.set_z_index(0)
            next_arrows.append(a)
        self.play(*[Create(a) for a in next_arrows], run_time=1.0)

        # pointers
        prev_ptr = Pointer("prev", YELLOW, y_offset=0.0)
        curr_ptr = Pointer("curr", GREEN,  y_offset=0.5)
        next_ptr = Pointer("next", BLUE,   y_offset=1.0)

        prev_ptr.place_instant(none_marker, "prev=None")
        curr_ptr.place_instant(nodes[0],    "curr")
        next_ptr.place_instant(nodes[1] if len(nodes) > 1 else none_marker, "next" if len(nodes) > 1 else "next=None")

        self.play(FadeIn(prev_ptr), FadeIn(curr_ptr), FadeIn(next_ptr), run_time=0.6)

        # caption
        caption = MarkupText("").scale(0.6).to_edge(DOWN)
        self.add(caption)

        code = MarkupText(
            "next = curr.next\ncurr.next = prev\nprev, curr = curr, next",
            font="Consolas"
        ).scale(0.4).to_corner(DL).set_opacity(0.85)
        bg = BackgroundRectangle(code, buff=0.15, corner_radius=0.12, fill_opacity=0.15)
        panel = VGroup(bg, code)
        self.add(panel)

        # algorithm state
        prev_idx, curr_idx = None, 0

        def flip_arrow(i, new_prev_idx):
            old = next_arrows[i]
            if new_prev_idx is None:
                arr = arrow_from_to(nodes[i].box, none_box, src_dir=LEFT, dst_dir=LEFT, z_index=-1)
            else:
                arr = arrow_from_to(nodes[i].box, nodes[new_prev_idx].box, src_dir=LEFT, dst_dir=RIGHT, z_index=-1)
            self.play(old.animate.set_color(ORANGE), run_time=0.25)
            self.play(ReplacementTransform(old, arr), run_time=0.6)
            return arr

        # iterate
        while curr_idx is not None:
            next_idx = curr_idx + 1 if curr_idx < len(values) - 1 else None

            set_caption("<span>1.</span> Save <span color='#4ea3ff'>next</span> = curr.next")
            self.play(nodes[curr_idx].box.animate.set_stroke(GREEN, 4).set_rate_func(there_and_back), run_time=0.6)

            # move next pointer
            move_and_pulse(
                next_ptr, 
                nodes[next_idx] if next_idx is not None else none_marker, 
                "next" if next_idx is not None else "next=None"
            )

            blink_line(next_idx)

            set_caption("<span>2.</span> Reverse link: curr.next = prev")
            next_arrows[curr_idx] = flip_arrow(curr_idx, prev_idx)

            set_caption("<span>3.</span> Move <span color='#ffd54a'>prev</span> = curr")
            move_and_pulse(
                prev_ptr, 
                nodes[curr_idx], 
                "prev"
            )

            blink_line(curr_idx)

            set_caption("<span>4.</span> Move <span color='#6de97c'>curr</span> = next")
            tgt = nodes[next_idx] if next_idx is not None else none_marker
            lbl = "curr" if next_idx is not None else "curr=None"
            move_and_pulse(
                curr_ptr, 
                tgt, 
                lbl
            )

            blink_line(next_idx)

            prev_idx, curr_idx = curr_idx, next_idx
            self.wait(pause * 0.6)
            if curr_idx is None:
                break

        set_caption("<span>Done:</span> New head is <span color='#ffd54a'>prev</span>.", t=1.2)
        # Keep this for 1-3 seconds then hide to show the next payoff caption

        brace = Brace(nodes, direction=UP, buff=0.3)
        final_text = Text("reversed").scale(0.6).next_to(brace, UP, buff=0.2)
        self.play(GrowFromCenter(brace), FadeIn(final_text, shift=UP*0.2), run_time=0.8)
        self.wait(2.3)

        self.wait(0.8)  # extra hold; 0.0–1.8 depending on how long you want
        self.play(caption.animate.set_opacity(0), run_time=0.2)

        # TODO Make this hidden so it looks like the code panel is moving here, rather than both code panel and below payoff_caption appearing at the same time and then overlapping into one.
        payoff_caption = MarkupText(
            "while curr: next=curr.next; curr.next=prev; prev=curr; curr=next",
            font="Consolas"
        ).scale(0.45).to_edge(DOWN)

        self.play(Transform(caption, payoff_caption), run_time=0.35)
        self.play(caption.animate.set_opacity(1), run_time=0.15)

        # Morph the code text into the payoff_caption and fade out the panel background
        self.play(
            FadeOut(bg, run_time=0.25),
            ReplacementTransform(code, caption),
        )

        # Optional: quick complexity caption
        complexity = Text("O(1) space · O(n) time").scale(0.4).next_to(caption, DOWN, buff=0.12)
        self.play(FadeIn(complexity, shift=UP*0.2), run_time=0.25)
        self.wait(0.7)

