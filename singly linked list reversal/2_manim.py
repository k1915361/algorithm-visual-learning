from manim import *

class TextAnimationDemo(Scene):
    def construct(self):
        # Step 1: Create text
        text1 = MarkupText("Step 1: Save next pointer")
        text1.scale(1.5).set_color(BLUE)
        self.play(FadeIn(text1))          # Fade in
        self.wait(1)                      # Pause

        # Step 2: Change text content
        text2 = MarkupText("Step 2: Reverse arrow direction")
        self.play(Transform(text1, text2))  # Transform into new text
        self.wait(1)

        # Step 3: Move text to bottom
        self.play(text1.animate.to_edge(DOWN))
        self.wait(1)

        # Step 4: Fade out
        self.play(FadeOut(text1))
        self.wait(0.5)

class NodeView(VGroup):
    def __init__(self, value, **kwargs):
        super().__init__(**kwargs)
        box = RoundedRectangle(
            corner_radius=0.15, 
            height=0.7, 
            width=0.8
        ).set_stroke(
            WHITE, 
            2
        )
        label = Text(str(value)).scale(0.8)
        label.move_to(box.get_center())
        self.box, self.label = box, label
        self.add(box, label)

    @property
    def mobj(self):
        return self.box

class Pointer(VGroup):
    def __init__(self, name, color, y_offset=0.0):
        super().__init__()
        self.name, self.color, self.y_offset = name, color, y_offset
        self.tag = Text(name).scale(0.7).set_color(color)
        self.dot = Dot(color=color)
        self.add(self.tag, self.dot)

    def _label_text(self, label_text):
        return (label_text if label_text is not None else self.name)

    def place_instant(self, ref_mobj, label_text=None, base_buff=0.45):
        """Position pointer immediately without animation (for initial layout)."""
        txt = Text(self._label_text(label_text)).scale(0.7).set_color(self.color)
        # become() replaces the glyphs in-place (keeps object identity)
        buff = base_buff + max(0, self.y_offset)        
        self.tag.become(txt)
        self.tag.next_to(ref_mobj, DOWN, buff=base_buff)
        self.tag.shift(DOWN * self.y_offset)
        self.dot.move_to(self.tag.get_bottom() + DOWN*0.12)

    def move_under_animations(self, ref_mobj, label_text=None, base_buff=0.45):
        """Return animations to move pointer under ref_mobj with an optional label."""
        target = Text(self._label_text(label_text)).scale(0.7).set_color(self.color)
        target.next_to(ref_mobj, DOWN, buff=base_buff)
        target.shift(DOWN * self.y_offset)
        return [
            Transform(self.tag, target),
            self.dot.animate.move_to(target.get_bottom() + DOWN*0.12),
        ]

    def pulse(self, scale_factor=1.2, run_time=0.2):
        # TODO . still same: the text only grows big and does not turn back.
        self.tag.save_state()
        self.dot.save_state()
        return Succession(
            AnimationGroup(
                self.tag.animate.scale(scale_factor),
                self.dot.animate.scale(scale_factor),
                lag_ratio=0.0,
            ).set_run_time(run_time),
            AnimationGroup(
                Restore(self.tag),
                Restore(self.dot),
                lag_ratio=0.0,
            ).set_run_time(run_time),
        )

class Theme:
    gap        = 2.0
    y_nodes    = 1.5
    fast       = 0.5
    pause      = 0.6

class Ports:
    RIGHT = RIGHT
    LEFT  = LEFT

def step_state(prev_idx, curr_idx, n):
    next_idx = curr_idx + 1 if curr_idx < n-1 else None
    return next_idx

# ------------------------------------------------------------
# Linked list reversal: cinematic pointer + arrow morphs (V2-lite)
# ------------------------------------------------------------
class LinkedListReverseScene(Scene):
    def construct(self):
        # ---------- Layout knobs ----------
        # values = [1, 2, 3, 4, 5]
        values = [1, 2, 3]
        gap = 2.0                     # horizontal spacing between nodes
        y_nodes = 1.5                 # y position for nodes
        y_labels = -1.2               # y for pointer labels
        arrow_buff = 0.25             # gap for next-pointer arrows
        pause = 0.6                   # pause between steps
        fast = 0.5                    # animation speed

        # ---------- Build nodes (rect + value) ----------
        nodes = []
        for i, v in enumerate(values):
            box = RoundedRectangle(corner_radius=0.15, height=0.7, width=1.0)
            box.set_stroke(WHITE, 2)
            box.move_to(LEFT*((len(values)-1)/2*gap) + RIGHT*i*gap + UP*y_nodes)

            label = Text(str(v)).scale(0.8).move_to(box.get_center())
            node = VGroup(box, label)
            nodes.append(node)
        nodes_group = VGroup(*nodes)

        # None marker on the right
        none_marker = Text("None").scale(0.7)
        none_marker.next_to(nodes[-1], RIGHT, buff=gap*0.6).align_to(nodes[-1], UP)
        none_box = SurroundingRectangle(none_marker, buff=0.08).set_opacity(0).set_stroke(width=0)
        self.add(none_box)

        self.play(*[FadeIn(n, shift=DOWN, lag_ratio=0.05) for n in nodes], FadeIn(none_marker), run_time=1.0)
        self.wait(0.4)

            # no need to be only vim. does sublime have a hotkey for quick search/jump and jump by word/paragraph/code-block and select with this jump? 
            # I was about to note a hotkey feature but I forgot

        def arrow_from_to(
            src_obj, dst_obj, *,
            src_dir=RIGHT, dst_dir=LEFT,
            pad=0.10,
            path_arc=0.0,          # NEW: bend amount (+ up, - down)
            z_index=None
        ):
            start = src_obj.get_edge_center(src_dir)
            end   = dst_obj.get_edge_center(dst_dir)
            arr = Arrow(
                start, end,
                buff=pad,
                stroke_width=6,
                max_tip_length_to_length_ratio=0.2,
                path_arc=path_arc,
            )
            if z_index is not None:
                arr.set_z_index(z_index)
            return arr


        # initial "next" arrows i -> i+1, last -> None
        next_arrows = []

        for i in range(len(values)):
            if i < len(values)-1:
                arr = arrow_from_to(
                    nodes[i][0], 
                    nodes[i+1][0], 
                    src_dir=RIGHT, 
                    dst_dir=LEFT,
                    path_arc=0.0,
                )
            else:
                arr = arrow_from_to(
                    nodes[i][0], 
                    none_box, 
                    src_dir=RIGHT, 
                    dst_dir=LEFT, 
                    pad=0.10,
                    path_arc=0.0,
                )
            next_arrows.append(arr)
        self.play(*[Create(a) for a in next_arrows], run_time=1.0)
        self.wait(0.3)

        for n in nodes:
            n.set_z_index(10)
        for a in next_arrows:
            a.set_z_index(0)
        none_marker.set_z_index(5)

        prev_ptr = Pointer("prev", YELLOW, y_offset=0.0)
        curr_ptr = Pointer("curr", GREEN,  y_offset=0.5)
        next_ptr = Pointer("next", BLUE,   y_offset=1.0)

        # place pointers initially
        prev_ptr[0].move_to(LEFT*5 + DOWN*abs(y_labels))  # off to the side
        prev_ptr[1].move_to(prev_ptr[0].get_bottom() + DOWN*0.15)

        prev_ptr.place_instant(none_marker, "prev=None")
        curr_ptr.place_instant(nodes[0],  "curr")

        if len(nodes) > 1:
            next_ptr.place_instant(nodes[1], "next")
        else:
            next_ptr.place_instant(none_marker, "next=None")

        self.play(FadeIn(prev_ptr), FadeIn(curr_ptr), FadeIn(next_ptr), run_time=0.6)

        # caption box
        caption = MarkupText("").scale(0.6).to_edge(DOWN)
        self.add(caption)

        # ---------- Algorithm state ----------
        prev_idx = None
        curr_idx = 0

        def set_caption(html, t=pause):
            caption.become(MarkupText(html).scale(0.6).to_edge(DOWN))
            self.wait(t)

        # highlight helper
        def highlight(i, color=GREEN):
            box = nodes[i][0]
            return box.animate.set_stroke(color, 4)

        def flip_arrow(i, new_prev_idx):
            old = next_arrows[i]
            if new_prev_idx is None:
                # First reversal for node i: point to the single right-side None
                # Use a bigger arc so it bends around the diagram
                arr = arrow_from_to(
                    nodes[i][0], none_box,
                    src_dir=LEFT, dst_dir=LEFT,   # entering None from its left side looks neat
                    pad=0.10,
                    path_arc=0.0,               # try +0.8 or -0.8 depending on where you want the bend
                    z_index=-1                   # optional: draw behind nodes
                )
            else:
                # Later reversals: point to previous node with a modest arc
                arr = arrow_from_to(
                    nodes[i][0], nodes[new_prev_idx][0],
                    src_dir=LEFT, dst_dir=RIGHT,
                    pad=0.10,
                    path_arc=0.0,              # small bend to avoid text
                    z_index=-1
                )

            self.play(old.animate.set_color(ORANGE), run_time=0.25)
            self.wait(0.1)
            self.play(ReplacementTransform(old, arr), run_time=0.6)
            self.play(old.animate.set_color(WHITE), run_time=0.1)
            return arr

        # ---------- Iterate the algorithm ----------
        # We'll collect any "null badges" so they don't get GC'd
        null_badges = [None]*len(values)

        while curr_idx is not None:
            # Compute next index (for labels) before flipping
            next_idx = curr_idx + 1 if curr_idx < len(values)-1 else None

            # Step A: Save next
            set_caption("<span>1.</span> Save <span color='#4ea3ff'>next</span> = curr.next")
            self.play(
                highlight(curr_idx, GREEN),
                run_time=fast
            )

            self.wait(1)

            self.play(
                highlight(curr_idx, WHITE),
                run_time=0.6
            )

            if next_idx is not None:
                self.play(
                    *next_ptr.move_under_animations(nodes[next_idx], "next"),        
                )
                self.play(
                    next_ptr.pulse(),    
                )
            else:
                self.play(
                    *next_ptr.move_under_animations(none_marker, "next=None"),
                )
                self.play(
                    next_ptr.pulse(),     
                )

            self.wait(0.3)

            # Step B: Reverse link (curr.next = prev)
            set_caption("<span>2. </span> Reverse link: curr.next = prev")
            new_arrow = flip_arrow(curr_idx, prev_idx)
            next_arrows[curr_idx] = new_arrow
            
            self.wait(0.3)

            # Step C: Move prev = curr
            set_caption("<span>3. </span> Move <span color='#ffd54a'>prev</span> = curr")
            
            self.play(
                *prev_ptr.move_under_animations(nodes[curr_idx], "prev"),
            )
            self.play(
                prev_ptr.pulse(),
            )


            self.wait(0.3)

            # Step D: Move curr = next
            set_caption("<span>4. </span> Move <span color='#6de97c'>curr</span> = next")
            target = nodes[next_idx] if next_idx is not None else none_marker
            label = "curr" if next_idx is not None else "curr=None"
            
            self.play(
                *curr_ptr.move_under_animations(target, label),
            )
            self.play(
                curr_ptr.pulse(),
            )

            # Advance indices
            prev_idx, curr_idx = curr_idx, next_idx

            # small pause each iteration
            self.wait(pause * 0.6)

            # End condition: once curr moves to None, stop
            if curr_idx is None:
                break

        set_caption("<span>Done:</span> New head is <span color='#ffd54a'>prev</span>.", t=1.2)

        # Optional: add a brace and label above the reversed chain
        brace = Brace(VGroup(*nodes), direction=UP, buff=0.3)
        final_text = Text("reversed").scale(0.6).next_to(brace, UP, buff=0.2)
        self.play(GrowFromCenter(brace), FadeIn(final_text, shift=UP*0.2), run_time=0.8)
        self.wait(1.3)
