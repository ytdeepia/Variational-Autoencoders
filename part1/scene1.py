from manim import *
from manim_voiceover import VoiceoverScene

import numpy as np
import matplotlib.pyplot as plt


class Scene1_1(VoiceoverScene):
    def construct(self):
        self.wait(2)

        # Generative AI vs traditional AI
        self.next_section(skip_animations=False)

        txt_genai = Text("Generative AI", color=WHITE)
        self.play(Write(txt_genai))
        self.wait(1)
        self.play(
            Flash(txt_genai, line_length=1.5, num_lines=20, flash_radius=2.5),
            run_time=1.5,
        )

        self.play(txt_genai.animate.to_edge(UP), run_time=1)

        openai_svg = SVGMobject("images/openai.svg")
        self.play(Create(openai_svg))
        self.play(openai_svg.animate.shift(LEFT * 3))

        llama_svg = SVGMobject("images/llama.svg")
        meta_svg = SVGMobject("images/meta.svg").scale(0.5)
        meta_svg.next_to(llama_svg, UP, buff=0.5)
        VGroup(llama_svg, meta_svg).move_to(ORIGIN).shift(RIGHT * 3)

        self.play(Create(llama_svg), Create(meta_svg), run_time=1.5)

        self.play(FadeOut(openai_svg, llama_svg, meta_svg))

        rect_txt = Text("Traditional AI", color=WHITE).scale(0.8)
        rect = SurroundingRectangle(rect_txt, buff=0.8, color=WHITE)

        input_txt = Text("Input", color=RED).next_to(rect, UP, buff=0.5).scale(0.8)
        input_rect = SurroundingRectangle(input_txt, buff=0.5, color=RED)
        VGroup(input_rect, input_txt).next_to(rect, LEFT, buff=1.5)

        output_txt = Text("Output", color=BLUE).next_to(rect, DOWN, buff=0.5).scale(0.8)
        output_rect = SurroundingRectangle(output_txt, buff=0.5, color=BLUE)
        VGroup(output_rect, output_txt).next_to(rect, RIGHT, buff=1.5)

        arrowin = Arrow(input_rect.get_right(), rect.get_left(), color=RED)
        arrowout = Arrow(rect.get_right(), output_rect.get_left(), color=BLUE)

        self.play(Create(rect), Write(rect_txt))
        self.play(Create(input_rect), Write(input_txt))
        self.play(GrowArrow(arrowin))
        self.play(GrowArrow(arrowout))
        self.play(Create(output_rect), Write(output_txt))

        self.play(
            FadeOut(rect_txt, arrowin, input_txt, input_rect),
            txt_genai.animate.move_to(rect_txt),
        )

        self.wait(1)

        self.wait(3)
        self.play(FadeOut(rect, output_rect, output_txt, arrowout, txt_genai))

        # VAE architecture
        self.next_section(skip_animations=False)

        encoder = Polygon(
            [-1, 1.4, 0], [1, 0.6, 0], [1, -0.6, 0], [-1, -1.4, 0], color=PURPLE
        )
        encoder_txt = Text("Encoder", color=WHITE).scale(0.6).move_to(encoder)

        mu = Rectangle(height=0.6, width=0.45, color=YELLOW)
        mu_txt = MathTex(r"\mu").scale(0.8).move_to(mu)
        sigma = Rectangle(height=0.6, width=0.45, color=YELLOW)
        sigma_txt = MathTex(r"\sigma").scale(0.8).move_to(sigma)

        mu = VGroup(mu, mu_txt)
        sigma = VGroup(sigma, sigma_txt)
        params = VGroup(mu, sigma).arrange(DOWN, buff=0.1).next_to(encoder, RIGHT, 0.25)

        distrib = Rectangle(height=1.2, width=1.8, color=RED).next_to(
            params, RIGHT, buff=0.25
        )
        distrib_txt = MathTex(r"\mathcal{N}(\mu, \sigma)").scale(0.8).move_to(distrib)
        distrib = VGroup(distrib, distrib_txt)

        sample = MathTex(r"\sim").scale(0.8).next_to(distrib, RIGHT, buff=0.2)

        z = Rectangle(height=1.2, width=0.45, color=BLUE).next_to(
            sample, RIGHT, buff=0.2
        )
        z_txt = MathTex(r"z").scale(0.8).move_to(z)
        z = VGroup(z, z_txt)

        decoder = Polygon(
            [-1, 0.6, 0], [1, 1.4, 0], [1, -1.4, 0], [-1, -0.6, 0], color=PURPLE
        )
        decoder.next_to(z, RIGHT, buff=0.2)
        decoder_txt = Text("Decoder", color=WHITE).scale(0.6).move_to(decoder)

        vae = VGroup(
            encoder, params, distrib, sample, z, decoder, encoder_txt, decoder_txt
        ).move_to(ORIGIN)

        vae_title = Text("Variational Autoencoder", color=WHITE).to_edge(UP)

        self.play(Write(vae_title))
        self.play(Create(vae), run_time=3)

        self.play(
            ShowPassingFlash(
                SurroundingRectangle(vae, buff=0.4, color=WHITE), time_width=0.4
            ),
            run_time=2,
        )

        self.wait(0.6)

        ae_title = Text("Why do we need that ?", color=WHITE).to_edge(UP)
        self.play(Transform(vae_title, ae_title))

        self.wait(0.4)

        # Autoencoder architecture
        self.next_section(skip_animations=False)

        ae_title = Text("Why not just autoencoders ?", color=WHITE).to_edge(UP)

        self.play(Transform(vae_title, ae_title))
        self.play(ShowPassingFlash(Underline(vae_title, color=WHITE), time_width=0.4))
        self.wait(1)

        self.play(
            LaggedStart(
                FadeOut(distrib, distrib_txt, sample, params),
                AnimationGroup(
                    z.animate.move_to(ORIGIN),
                    VGroup(encoder, encoder_txt).animate.move_to(1.5 * LEFT),
                    VGroup(decoder, decoder_txt).animate.move_to(1.5 * RIGHT),
                ),
                lag_ratio=0.5,
            ),
            run_time=2,
        )

        self.play(FadeOut(vae_title, z, encoder, decoder, encoder_txt, decoder_txt))

        self.wait(2)


if __name__ == "__main__":
    scene = Scene1_1()
    scene.render()
