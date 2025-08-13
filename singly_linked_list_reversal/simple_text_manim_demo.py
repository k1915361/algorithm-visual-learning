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