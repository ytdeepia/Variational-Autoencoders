from manim import *
from manim_voiceover import VoiceoverScene
from manim_voiceover.services.elevenlabs import ElevenLabsService

import numpy as np
import matplotlib.pyplot as plt


class Scene2_1(VoiceoverScene):
    def construct(self):

        self.wait(2)

        txt = Tex("Who is behind VAEs?").scale(1.5)

        self.play(Write(txt))

        self.play(txt.animate.to_edge(UP))

        kingma_img = ImageMobject("images/kingma.jpg").scale(1)
        kingma_rect = SurroundingRectangle(kingma_img, color=WHITE, buff=0)
        kingma_txt = Tex("Diederik P. Kingma", color=WHITE).next_to(
            kingma_img, UP, buff=0.5
        )
        kingma_txt_ul = Underline(kingma_txt)

        paper_img = (
            ImageMobject("images/vae_paper.png").scale(0.4).to_edge(RIGHT, buff=1.5)
        )

        self.play(FadeIn(kingma_img, kingma_rect, kingma_txt), FadeOut(txt))
        self.play(Create(kingma_txt_ul))
        self.play(
            Group(kingma_img, kingma_txt, kingma_rect, kingma_txt_ul).animate.to_edge(
                LEFT, buff=1.5
            )
        )
        self.play(FadeIn(paper_img))

        self.play(FadeOut(paper_img))
        adam_txt = (
            Tex("Adam Optimizer", color=WHITE)
            .to_edge(RIGHT, buff=2.0)
            .to_edge(UP, buff=1.0)
        )
        adam_txt_ul = Underline(adam_txt)
        self.play(FadeIn(adam_txt), GrowFromEdge(adam_txt_ul, UP))

        # Define the curve (with multiple local minima)
        def func(x):
            return 0.1 * (
                x**4 - 6 * x**2 + 4 * x
            )  # Example function with multiple local minima

        axes = (
            Axes(
                x_range=[-4, 4, 1],
                y_range=[-4, 4, 1],
                axis_config={"include_numbers": True},
            )
            .scale(0.4)
            .next_to(adam_txt, DOWN, buff=4)
        )

        curve = axes.plot(func, x_range=[-4, 4], color=BLUE)
        dot = Dot(axes.c2p(-3.5, func(-3.5)), radius=0.15, color=RED, fill_opacity=0.8)
        self.play(Create(curve))
        self.play(FadeIn(dot))

        # Define the path for gradient descent (go a little beyond the local minimum)
        descent_path = axes.plot(func, x_range=[-3.5, -0.2], color=BLUE)
        oscillation_path1 = axes.plot(func, x_range=[-2.6, -0.2], color=BLUE)
        oscillation_path2 = axes.plot(func, x_range=[-2.6, -0.9], color=BLUE)
        oscillation_path3 = axes.plot(func, x_range=[-2.3, -0.9], color=BLUE)
        oscillation_path4 = axes.plot(func, x_range=[-2.3, -1.8], color=BLUE)

        # Move the dot along the curve towards the local minimum and a bit beyond
        self.play(MoveAlongPath(dot, descent_path, rate_func=smooth, run_time=1.5))
        self.play(
            MoveAlongPath(
                dot,
                oscillation_path1,
                rate_func=lambda t: smooth(1 - t),
                run_time=1.5,
            ),
        )
        self.play(MoveAlongPath(dot, oscillation_path2, rate_func=smooth, run_time=1.5))
        self.play(
            MoveAlongPath(
                dot,
                oscillation_path3,
                rate_func=lambda t: smooth(1 - t),
                run_time=1.5,
            ),
        )
        self.play(
            MoveAlongPath(dot, oscillation_path4, rate_func=smooth, run_time=1.5),
        )
        self.play(
            FadeOut(
                kingma_img,
                kingma_txt,
                kingma_rect,
                kingma_txt_ul,
                adam_txt,
                adam_txt_ul,
                curve,
                dot,
            )
        )

        txt_dl = (
            Tex("Deep Learning", color=BLUE)
            .scale(1.5)
            .set_color_by_gradient(BLUE_D, BLUE_B)
        )
        txt_proba = (
            Tex("Bayesian statistics", color=RED)
            .scale(1.5)
            .set_color_by_gradient(RED_D, RED_B)
        )

        VGroup(txt_dl, txt_proba).arrange(RIGHT, buff=1).move_to(ORIGIN)

        self.play(Write(txt_dl))
        self.play(ShowPassingFlash(SurroundingRectangle(txt_dl, color=BLUE)))
        self.play(Write(txt_proba))
        self.play(ShowPassingFlash(SurroundingRectangle(txt_proba, color=RED)))

        self.play(FadeOut(txt_dl, txt_proba))

        self.wait(2)


if __name__ == "__main__":
    scene = Scene2_1()
    scene.render()
