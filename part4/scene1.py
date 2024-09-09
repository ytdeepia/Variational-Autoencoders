from manim import *
from manim_voiceover import VoiceoverScene

import numpy as np
import matplotlib.pyplot as plt


class Scene4_1(VoiceoverScene):
    def construct(self):

        self.wait(2)

        # Blurry reconstructions
        self.next_section(skip_animations=False)

        txt = Text("VAEs produce blurry images")

        self.play(Write(txt))

        self.wait(0.8)

        self.play(FadeOut(txt))

        ref0 = ImageMobject("./images/recon/ref_0.png").scale(2)
        ref0_rect = SurroundingRectangle(ref0, buff=0).set_stroke(color=WHITE)
        ref1 = ImageMobject("./images/recon/ref_1.png").scale(2)
        ref1_rect = SurroundingRectangle(ref1, buff=0).set_stroke(color=WHITE)
        ref2 = ImageMobject("./images/recon/ref_2.png").scale(2)
        ref2_rect = SurroundingRectangle(ref2, buff=0).set_stroke(color=WHITE)
        ref3 = ImageMobject("./images/recon/ref_3.png").scale(2)
        ref3_rect = SurroundingRectangle(ref3, buff=0).set_stroke(color=WHITE)

        recon0 = ImageMobject("./images/recon/recon_0.png").scale(2)
        recon0_rect = SurroundingRectangle(recon0, buff=0).set_stroke(color=WHITE)
        recon1 = ImageMobject("./images/recon/recon_1.png").scale(2)
        recon1_rect = SurroundingRectangle(recon1, buff=0).set_stroke(color=WHITE)
        recon2 = ImageMobject("./images/recon/recon_2.png").scale(2)
        recon2_rect = SurroundingRectangle(recon2, buff=0).set_stroke(color=WHITE)
        recon3 = ImageMobject("./images/recon/recon_3.png").scale(2)
        recon3_rect = SurroundingRectangle(recon3, buff=0).set_stroke(color=WHITE)

        title = Text("CelebA dataset").scale(0.8).to_edge(UP)
        title_ul = Underline(title)

        Group(ref0, ref0_rect).shift(1.25 * (UP + LEFT))
        Group(ref1, ref1_rect).shift(1.25 * (UP + RIGHT))
        Group(ref2, ref2_rect).shift(1.25 * (DOWN + RIGHT))
        Group(ref3, ref3_rect).shift(1.25 * (DOWN + LEFT))

        self.play(
            LaggedStart(Write(title), GrowFromEdge(title_ul, LEFT), lag_ratio=0.7)
        )
        self.play(GrowFromPoint(Group(ref0, ref0_rect), ORIGIN))
        self.play(GrowFromPoint(Group(ref1, ref1_rect), ORIGIN))
        self.play(GrowFromPoint(Group(ref2, ref2_rect), ORIGIN))
        self.play(GrowFromPoint(Group(ref3, ref3_rect), ORIGIN))

        self.wait(0.7)

        self.play(
            Group(
                title,
                title_ul,
                ref0,
                ref1,
                ref2,
                ref3,
                ref0_rect,
                ref1_rect,
                ref2_rect,
                ref3_rect,
            ).animate.shift((config.frame_width / 4) * LEFT)
        )

        title_reconstructions = (
            Text("Reconstructions")
            .scale(0.8)
            .to_edge(UP)
            .shift((config.frame_width / 4) * RIGHT)
        )
        title_reconstructions_ul = Underline(title_reconstructions)

        self.play(
            LaggedStart(
                Write(title_reconstructions),
                GrowFromEdge(title_reconstructions_ul, LEFT),
                lag_ratio=0.7,
            )
        )

        Group(recon0, recon0_rect).shift(1.25 * (UP + LEFT))
        Group(recon1, recon1_rect).shift(1.25 * (UP + RIGHT))
        Group(recon2, recon2_rect).shift(1.25 * (DOWN + RIGHT))
        Group(recon3, recon3_rect).shift(1.25 * (DOWN + LEFT))

        Group(
            recon0,
            recon0_rect,
            recon1,
            recon1_rect,
            recon2,
            recon2_rect,
            recon3,
            recon3_rect,
        ).shift((config.frame_width / 4) * RIGHT)

        self.play(FadeIn(Group(recon0, recon0_rect)))
        self.play(FadeIn(Group(recon1, recon1_rect)))
        self.play(FadeIn(Group(recon2, recon2_rect)))
        self.play(FadeIn(Group(recon3, recon3_rect)))

        self.play(
            FadeOut(
                title,
                title_ul,
                title_reconstructions,
                title_reconstructions_ul,
                ref0,
                ref1,
                ref2,
                ref3,
                ref0_rect,
                ref1_rect,
                ref2_rect,
                ref3_rect,
                recon0,
                recon0_rect,
                recon1,
                recon1_rect,
                recon2,
                recon2_rect,
                recon3,
                recon3_rect,
            )
        )

        title = Text("VAE Samples").scale(0.7).to_edge(UP)
        title_ul = Underline(title)

        self.play(
            LaggedStart(Write(title), GrowFromEdge(title_ul, LEFT), lag_ratio=0.7)
        )

        sample1 = ImageMobject("./images/samples/sample_0.png").scale_to_fit_width(1.5)
        sample1_rect = SurroundingRectangle(sample1, buff=0).set_stroke(color=WHITE)
        sample2 = ImageMobject("./images/samples/sample_1.png").scale_to_fit_width(1.5)
        sample2_rect = SurroundingRectangle(sample2, buff=0).set_stroke(color=WHITE)
        sample3 = ImageMobject("./images/samples/sample_2.png").scale_to_fit_width(1.5)
        sample3_rect = SurroundingRectangle(sample3, buff=0).set_stroke(color=WHITE)
        sample4 = ImageMobject("./images/samples/sample_3.png").scale_to_fit_width(1.5)
        sample4_rect = SurroundingRectangle(sample4, buff=0).set_stroke(color=WHITE)

        Group(sample1, sample1_rect).shift(1 * (UP + LEFT))
        Group(sample2, sample2_rect).shift(1 * (UP + RIGHT))
        Group(sample3, sample3_rect).shift(1 * (DOWN + RIGHT))
        Group(sample4, sample4_rect).shift(1 * (DOWN + LEFT))

        self.play(GrowFromPoint(Group(sample1, sample1_rect), ORIGIN))
        self.play(GrowFromPoint(Group(sample2, sample2_rect), ORIGIN))
        self.play(GrowFromPoint(Group(sample3, sample3_rect), ORIGIN))
        self.play(GrowFromPoint(Group(sample4, sample4_rect), ORIGIN))

        self.play(
            Group(
                title,
                title_ul,
                sample1,
                sample2,
                sample3,
                sample4,
                sample1_rect,
                sample2_rect,
                sample3_rect,
                sample4_rect,
            ).animate.shift((config.frame_width / 3) * LEFT)
        )

        title_gan = Text("GAN Samples").scale(0.7).to_edge(UP)
        title_gan_ul = Underline(title_gan)
        self.play(
            LaggedStart(
                Write(title_gan), GrowFromEdge(title_gan_ul, LEFT), lag_ratio=0.7
            )
        )

        sample_gan1 = ImageMobject("./images/gan/sample0.png").scale_to_fit_width(1.5)
        sample_gan1_rect = SurroundingRectangle(sample_gan1, buff=0).set_stroke(
            color=WHITE
        )
        sample_gan2 = ImageMobject("./images/gan/sample1.png").scale_to_fit_width(1.5)
        sample_gan2_rect = SurroundingRectangle(sample_gan2, buff=0).set_stroke(
            color=WHITE
        )
        sample_gan3 = ImageMobject("./images/gan/sample2.png").scale_to_fit_width(1.5)
        sample_gan3_rect = SurroundingRectangle(sample_gan3, buff=0).set_stroke(
            color=WHITE
        )
        sample_gan4 = ImageMobject("./images/gan/sample3.png").scale_to_fit_width(1.5)
        sample_gan4_rect = SurroundingRectangle(sample_gan4, buff=0).set_stroke(
            color=WHITE
        )

        Group(sample_gan1, sample_gan1_rect).shift(1 * (UP + LEFT))
        Group(sample_gan2, sample_gan2_rect).shift(1 * (UP + RIGHT))
        Group(sample_gan3, sample_gan3_rect).shift(1 * (DOWN + RIGHT))
        Group(sample_gan4, sample_gan4_rect).shift(1 * (DOWN + LEFT))

        self.play(GrowFromPoint(Group(sample_gan1, sample_gan1_rect), ORIGIN))
        self.play(GrowFromPoint(Group(sample_gan2, sample_gan2_rect), ORIGIN))
        self.play(GrowFromPoint(Group(sample_gan3, sample_gan3_rect), ORIGIN))
        self.play(GrowFromPoint(Group(sample_gan4, sample_gan4_rect), ORIGIN))

        self.play(
            Group(
                title_gan,
                title_gan_ul,
                sample_gan1,
                sample_gan2,
                sample_gan3,
                sample_gan4,
                sample_gan1_rect,
                sample_gan2_rect,
                sample_gan3_rect,
                sample_gan4_rect,
            ).animate.shift((config.frame_width / 3) * RIGHT)
        )

        title_diff = Text("Diffusion samples").scale(0.7).to_edge(UP)
        title_diff_ul = Underline(title_diff)
        self.play(
            LaggedStart(
                Write(title_diff), GrowFromEdge(title_diff_ul, LEFT), lag_ratio=0.7
            )
        )

        sample_diff1 = ImageMobject(
            "./images/diffusion/sample0.png"
        ).scale_to_fit_width(1.5)
        sample_diff1_rect = SurroundingRectangle(sample_diff1, buff=0).set_stroke(
            color=WHITE
        )
        sample_diff2 = ImageMobject(
            "./images/diffusion/sample1.png"
        ).scale_to_fit_width(1.5)
        sample_diff2_rect = SurroundingRectangle(sample_diff2, buff=0).set_stroke(
            color=WHITE
        )
        sample_diff3 = ImageMobject(
            "./images/diffusion/sample2.png"
        ).scale_to_fit_width(1.5)
        sample_diff3_rect = SurroundingRectangle(sample_diff3, buff=0).set_stroke(
            color=WHITE
        )
        sample_diff4 = ImageMobject(
            "./images/diffusion/sample3.png"
        ).scale_to_fit_width(1.5)
        sample_diff4_rect = SurroundingRectangle(sample_diff4, buff=0).set_stroke(
            color=WHITE
        )
        Group(sample_diff1, sample_diff1_rect).shift(1 * (UP + LEFT))
        Group(sample_diff2, sample_diff2_rect).shift(1 * (UP + RIGHT))
        Group(sample_diff3, sample_diff3_rect).shift(1 * (DOWN + RIGHT))
        Group(sample_diff4, sample_diff4_rect).shift(1 * (DOWN + LEFT))

        self.play(GrowFromPoint(Group(sample_diff1, sample_diff1_rect), ORIGIN))
        self.play(GrowFromPoint(Group(sample_diff2, sample_diff2_rect), ORIGIN))
        self.play(GrowFromPoint(Group(sample_diff3, sample_diff3_rect), ORIGIN))
        self.play(GrowFromPoint(Group(sample_diff4, sample_diff4_rect), ORIGIN))

        # New models
        self.next_section(skip_animations=False)

        self.play(
            FadeOut(
                title,
                title_gan,
                title_diff,
                title_ul,
                title_gan_ul,
                title_diff_ul,
                sample1,
                sample2,
                sample3,
                sample4,
                sample1_rect,
                sample2_rect,
                sample3_rect,
                sample4_rect,
                sample_gan1,
                sample_gan2,
                sample_gan3,
                sample_gan4,
                sample_gan1_rect,
                sample_gan2_rect,
                sample_gan3_rect,
                sample_gan4_rect,
                sample_diff1,
                sample_diff2,
                sample_diff3,
                sample_diff4,
                sample_diff1_rect,
                sample_diff2_rect,
                sample_diff3_rect,
                sample_diff4_rect,
            )
        )

        txt = Text("How do we even choose which image to generate?").scale(0.6)

        self.play(Write(txt))

        self.play(FadeOut(txt))
        cvaepaper = (
            ImageMobject("./images/papers/cvae.jpg")
            .scale(0.4)
            .shift(4 * LEFT + 8 * DOWN)
        )
        betavaepaper = (
            ImageMobject("./images/papers/betavae.jpg").scale(0.4).shift(8 * DOWN)
        )
        vqvaepaper = (
            ImageMobject("./images/papers/vqvae.jpg")
            .scale(0.4)
            .shift(4 * RIGHT + 8 * DOWN)
        )
        self.add(cvaepaper, betavaepaper, vqvaepaper)
        self.play(
            LaggedStart(
                cvaepaper.animate.shift(8 * UP),
                betavaepaper.animate.shift(8 * UP),
                vqvaepaper.animate.shift(8 * UP),
                lag_ratio=0.5,
            ),
            run_time=3,
        )

        cvae = Tex("CVAE").scale(0.8)
        cvae_rect = SurroundingRectangle(cvae, buff=0.3, color=WHITE)
        VGroup(cvae, cvae_rect).next_to(cvaepaper, UP, buff=0.5)

        betavae = MathTex(r"\beta \text{-VAE}").scale(0.8)
        betavae_rect = SurroundingRectangle(betavae, buff=0.3, color=WHITE)
        VGroup(betavae, betavae_rect).next_to(betavaepaper, UP, buff=0.5)

        vqvae = Tex("VQ-VAE").scale(0.8)
        vqvae_rect = SurroundingRectangle(vqvae, buff=0.3, color=WHITE)
        VGroup(vqvae, vqvae_rect).next_to(vqvaepaper, UP, buff=0.5)

        self.play(LaggedStart(Write(cvae), Create(cvae_rect), lag_ratio=0.8))
        self.wait(5)
        self.play(LaggedStart(Write(betavae), Create(betavae_rect), lag_ratio=0.8))
        self.wait(5)
        self.play(LaggedStart(Write(vqvae), Create(vqvae_rect), lag_ratio=0.8))

        self.wait(0.6)

        self.play(FadeOut(cvae, cvae_rect, betavae, betavae_rect, vqvae, vqvae_rect))

        self.wait(2)


if __name__ == "__main__":
    scene = Scene4_1()
    scene.render()
