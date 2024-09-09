from manim import *
from manim_voiceover import VoiceoverScene


class Scene2_3(ThreeDScene, VoiceoverScene):
    def construct(self):

        self.wait(2)

        # Show p(x) and samples
        self.next_section(skip_animations=False)

        data_dist = SVGMobject(
            "images/data_dist.svg",
            fill_color=[RED_C, RED_E],
            fill_opacity=1,
            stroke_opacity=0,
        ).scale(1.5)
        data_title = (
            Tex("Data Distribution", color=RED_D)
            .scale(0.6)
            .next_to(data_dist, DOWN, buff=0.5)
        )
        data_symbol = (
            MathTex(r"p(x)", color=RED_D).scale(1.2).next_to(data_dist, UP, buff=0.4)
        )
        data_title_ul = Underline(data_title, color=RED_D)

        self.play(Create(data_dist), Write(data_symbol))
        self.play(
            LaggedStart(
                Write(data_title), GrowFromEdge(data_title_ul, LEFT), lag_ratio=0.6
            )
        )

        self.wait(2)

        dot1 = Dot(color=WHITE).move_to(data_dist.get_center() + 0.2 * (UP + RIGHT))
        dot1_label = (
            MathTex("x_1").scale(0.8).next_to(dot1, UP, buff=0.1).set_color(WHITE)
        )
        sample1 = (
            ImageMobject("images/ref_0.png")
            .set_resampling_algorithm(RESAMPLING_ALGORITHMS["none"])
            .scale(8)
            .next_to(data_dist, RIGHT, buff=1)
        )
        sample1_rect = SurroundingRectangle(sample1, buff=0.1, color=WHITE)

        self.play(Create(dot1), Write(dot1_label))
        self.play(GrowFromPoint(Group(sample1, sample1_rect), dot1.get_center()))

        dot2 = Dot(color=WHITE).move_to(data_dist.get_center() + 0.3 * (DOWN + LEFT))
        dot2_label = (
            MathTex("x_2").scale(0.8).next_to(dot2, DOWN, buff=0.1).set_color(WHITE)
        )
        sample2 = (
            ImageMobject("images/ref_1.png")
            .set_resampling_algorithm(RESAMPLING_ALGORITHMS["none"])
            .scale(8)
            .next_to(data_dist, LEFT, buff=1)
        )
        sample2_rect = SurroundingRectangle(sample2, buff=0.1, color=WHITE)

        self.play(Create(dot2), Write(dot2_label))
        self.play(GrowFromPoint(Group(sample2, sample2_rect), dot2.get_center()))

        self.wait(0.6)

        self.play(
            LaggedStart(
                FadeOut(
                    dot1,
                    dot1_label,
                    sample1,
                    sample1_rect,
                    dot2,
                    dot2_label,
                    sample2,
                    sample2_rect,
                ),
                VGroup(
                    data_dist, data_title, data_title_ul, data_symbol
                ).animate.to_edge(LEFT, buff=1),
                lag_ratio=0.6,
            )
        )

        z_space = (
            Tex("Latent Space")
            .shift(config.frame_width * RIGHT / 4)
            .scale(0.8)
            .to_edge(UP, buff=0.2)
        )
        z_space_ul = Underline(z_space)

        latent_dist = (
            SVGMobject(
                "images/latent_dist.svg",
                fill_color=[BLUE_C, BLUE_E],
                fill_opacity=1,
                stroke_opacity=0,
            )
            .scale(1.5)
            .to_edge(RIGHT, buff=1)
        )
        latent_title = (
            Tex("Latent Distribution", color=BLUE_D)
            .scale(0.6)
            .next_to(latent_dist, DOWN, buff=0.5)
        )
        latent_symbol = (
            MathTex(r"p(z)", color=BLUE_D).scale(1.2).next_to(latent_dist, UP, buff=0.4)
        )
        latent_title_ul = Underline(latent_title, color=BLUE_D)

        middle_line = DashedLine(
            UP * config.frame_height / 2, DOWN * config.frame_height / 2, buff=0
        )

        self.play(Create(latent_dist), Write(latent_symbol))
        self.play(
            LaggedStart(
                Write(latent_title),
                GrowFromEdge(latent_title_ul, LEFT),
                lag_ratio=0.6,
            )
        )

        self.wait()
        self.play(Create(middle_line), run_time=2)
        self.wait(1)
        self.play(FadeOut(middle_line))

        self.wait(0.9)

        # Mappings
        self.next_section(skip_animations=False)

        posterior_arrow = CurvedArrow(
            LEFT + UP,
            RIGHT + UP,
            angle=-PI / 2,
        )
        posterior = MathTex(r"p(z|x)").next_to(posterior_arrow, UP, buff=0.3)

        likelihood_arrow = CurvedArrow(
            DOWN + RIGHT,
            DOWN + LEFT,
            angle=-PI / 2,
        )
        likelihood = MathTex(r"p(x|z)").next_to(likelihood_arrow, DOWN, buff=0.3)

        self.wait(1)
        self.play(Create(posterior_arrow), run_time=1.5)

        self.play(Write(posterior))
        self.play(
            ShowPassingFlash(
                SurroundingRectangle(posterior, buff=0.1, color=WHITE),
                time_width=0.6,
            ),
            run_time=1.5,
        )

        self.wait(0.7)

        self.wait(1)
        self.play(Create(likelihood_arrow), run_time=1.5)
        self.play(Write(likelihood))
        self.play(
            ShowPassingFlash(
                SurroundingRectangle(likelihood, buff=0.1, color=WHITE),
                time_width=0.6,
            ),
            run_time=1.5,
        )

        self.play(
            Indicate(posterior, color=posterior.color, scale_factor=1.5),
            run_time=1.5,
        )
        self.wait(6)
        self.play(
            Indicate(data_symbol, color=data_symbol.color, scale_factor=1.5),
            run_time=1.5,
        )

        self.play(
            Flash(
                data_dist,
                color=data_symbol.color,
                flash_radius=1,
                line_length=0.8,
                num_lines=24,
            ),
            run_time=3,
        )

        # Normal distribution for the latent
        self.next_section(skip_animations=False)

        self.wait(4)
        rect = SurroundingRectangle(latent_symbol, buff=0.1, color=latent_symbol.color)
        self.play(ShowPassingFlash(rect), run_time=2)

        sigma = 2
        num = 150
        colors = color_gradient([BLUE_D, BLACK], num)
        normal = VGroup(
            *[
                Circle(
                    radius=(i + 1) * sigma / num,
                    stroke_width=2 * sigma,
                    color=colors[i],
                )
                for i in range(num)
            ]
        ).move_to(latent_dist)

        latent_symbol_normal = MathTex(
            r"p(z) = \mathcal{N}(0, 1)", color=latent_symbol.color
        ).move_to(latent_symbol)

        self.play(Transform(latent_dist, normal), run_time=3)
        self.play(Transform(latent_symbol, latent_symbol_normal))

        # Posterior as a Gaussian
        self.next_section(skip_animations=False)

        rect = SurroundingRectangle(likelihood, buff=0.1, color=WHITE)
        self.play(ShowPassingFlash(rect), run_time=2)

        rect = SurroundingRectangle(posterior, buff=0.1, color=WHITE)
        self.play(ShowPassingFlash(rect), run_time=2)

        posterior_approx = MathTex(r"q(z | x) \approx p(z | x)").move_to(posterior)
        self.play(Transform(posterior, posterior_approx), run_time=1)

        posterior_gaussian = MathTex(
            r"q(z | x) = \mathcal{N}(", r"\mu", ",", r"\sigma", ")"
        ).move_to(posterior)

        self.play(Transform(posterior, posterior_gaussian), run_time=1)

        rect = SurroundingRectangle(posterior[1], buff=0.1, color=YELLOW)
        self.play(Create(rect), run_time=1)
        self.wait(1)
        rect2 = SurroundingRectangle(posterior[3], buff=0.1, color=YELLOW)
        self.play(Transform(rect, rect2), run_time=1)

        txt_bayes = (
            Tex("Variational Bayes", color=WHITE).scale(1.2).to_edge(UP, buff=0.2)
        )
        txt_bayes_ul = Underline(txt_bayes)

        self.play(
            FadeOut(rect),
            LaggedStart(
                Write(txt_bayes), GrowFromEdge(txt_bayes_ul, LEFT), lag_ratio=0.6
            ),
        )

        # Introducing the autoencoder
        self.next_section(skip_animations=False)

        encoder = Polygon(
            [-1, 1.4, 0], [1, 0.6, 0], [1, -0.6, 0], [-1, -1.4, 0], color=PURPLE
        )
        encoder_txt = Text("Encoder", color=WHITE).scale(0.6).move_to(encoder)

        VGroup(encoder, encoder_txt).scale(0.8).move_to(posterior_arrow.get_center())

        self.play(FadeOut(posterior, posterior_arrow))
        self.play(Create(encoder), Write(encoder_txt))

        txt_bayes_auto = (
            Tex("Autoencoder Variational Bayes", color=WHITE)
            .scale(1.2)
            .to_edge(UP, buff=0.2)
        )
        txt_bayes_auto_ul = Underline(txt_bayes_auto)

        self.play(LaggedStart(FadeOut(txt_bayes, txt_bayes_ul), Write(txt_bayes_auto)))
        self.play(GrowFromEdge(txt_bayes_auto_ul, LEFT))

        decoder = Polygon(
            [-1, 1.4, 0], [1, 0.6, 0], [1, -0.6, 0], [-1, -1.4, 0], color=PURPLE
        )
        decoder_txt = Text("Decoder", color=WHITE).scale(0.6).move_to(decoder)

        VGroup(decoder, decoder_txt).scale(0.8).move_to(likelihood_arrow.get_center())

        self.play(FadeOut(likelihood, likelihood_arrow))
        self.play(Create(decoder), Write(decoder_txt))

        self.play(
            FadeOut(
                txt_bayes_auto,
                txt_bayes_auto_ul,
                encoder,
                decoder,
                encoder_txt,
                decoder_txt,
                latent_dist,
                latent_symbol,
                latent_title,
                latent_title_ul,
                data_dist,
                data_symbol,
                data_title,
                data_title_ul,
            )
        )

        txt = Text("How do we train this autoencoder ?", color=WHITE).scale(1.2)

        self.play(Write(txt))

        self.play(FadeOut(txt))

        self.wait(2)


if __name__ == "__main__":
    scene = Scene2_3()
    scene.render()
