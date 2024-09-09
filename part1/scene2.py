from manim import *
from manim_voiceover import VoiceoverScene


class Scene1_2(VoiceoverScene):
    def construct(self):

        self.wait(2)

        # Autoencoder principle
        self.next_section(skip_animations=False)

        encoder = Polygon(
            [-1, 1.4, 0], [1, 0.6, 0], [1, -0.6, 0], [-1, -1.4, 0], color=PURPLE
        )
        encoder_txt = Text("Encoder", color=WHITE).scale(0.6).move_to(encoder)

        z = Rectangle(height=1.2, width=0.45, color=BLUE).next_to(
            encoder, RIGHT, buff=0.2
        )
        z_txt = MathTex(r"z").scale(0.8).move_to(z)
        z = VGroup(z, z_txt)

        decoder = Polygon(
            [-1, 0.6, 0], [1, 1.4, 0], [1, -1.4, 0], [-1, -0.6, 0], color=PURPLE
        )
        decoder.next_to(z, RIGHT, buff=0.2)
        decoder_txt = Text("Decoder", color=WHITE).scale(0.6).move_to(decoder)

        ae = VGroup(encoder, encoder_txt, z, decoder, decoder_txt).move_to(ORIGIN)

        self.play(FadeIn(ae), run_time=2)

        img_ref = (
            ImageMobject("images/reconstructions/ref_3.png")
            .set_resampling_algorithm(RESAMPLING_ALGORITHMS["none"])
            .scale(8)
            .next_to(ae, LEFT, buff=2)
        )

        img_recon = (
            ImageMobject("images/reconstructions/reconstruction_3.png")
            .scale(8)
            .set_resampling_algorithm(RESAMPLING_ALGORITHMS["none"])
            .next_to(ae, RIGHT, buff=2)
        )

        img_ref_rect = SurroundingRectangle(img_ref, color=WHITE, buff=0.1)
        img_recon_rect = SurroundingRectangle(img_recon, color=WHITE, buff=0.1)

        arrowin = Arrow(img_ref.get_right(), ae.get_left(), color=WHITE)
        arrowout = Arrow(ae.get_right(), img_recon.get_left(), color=WHITE)

        self.wait(1)
        self.play(Create(img_ref_rect), FadeIn(img_ref))
        self.play(GrowArrow(arrowin))
        self.wait(1)
        self.play(GrowArrow(arrowout))
        self.play(Create(img_recon_rect), FadeIn(img_recon))

        self.play(
            Group(
                img_ref_rect,
                img_recon_rect,
                img_recon,
                img_ref,
                arrowin,
                arrowout,
                ae,
            ).animate.to_edge(UP)
        )

        latent_vector = MathTex(
            r"\begin{bmatrix} 0.23 \\ \\ \hdots \\ \\ 0.63 \end{bmatrix}",
            color=BLUE,
        ).next_to(ae, DOWN, buff=0.5)
        latent_vector_txt = Text("Latent representation", color=WHITE).scale(0.6)
        latent_vector_txt_rect = SurroundingRectangle(latent_vector_txt, color=WHITE)

        VGroup(latent_vector_txt, latent_vector_txt_rect).next_to(
            latent_vector, LEFT, buff=0.5
        )
        arrow_latent = Arrow(z.get_bottom(), latent_vector.get_top(), color=BLUE)

        self.play(GrowArrow(arrow_latent))
        self.play(Create(latent_vector))
        self.play(
            LaggedStart(
                Create(latent_vector_txt_rect),
                Create(latent_vector_txt),
                lag_ratio=0.5,
            )
        )

        self.wait(0.6)

        latent_axes = Axes(
            x_range=[0, 1, 1],
            y_range=[0, 1, 1],
            x_length=3,
            y_length=3,
            axis_config={
                "color": WHITE,
                "include_tip": True,
                "include_ticks": False,
            },
        ).next_to(ae, DOWN, buff=0.5)

        latent_axes_txt = Text("Latent space", color=WHITE).scale(0.6)
        latent_axes_txt_rect = SurroundingRectangle(latent_axes_txt, color=WHITE)
        VGroup(latent_axes_txt, latent_axes_txt_rect).next_to(
            latent_axes, LEFT, buff=0.5
        )

        self.play(FadeOut(latent_vector, latent_vector_txt, latent_vector_txt_rect))
        self.play(FadeIn(latent_axes))
        self.play(
            LaggedStart(
                Create(latent_axes_txt_rect),
                Create(latent_axes_txt),
                lag_ratio=0.5,
            )
        )

        dot = Dot(latent_axes.c2p(0.23, 0.63), color=BLUE)
        dot_txt = (
            MathTex(r"z", color=BLUE).scale(0.8).next_to(dot, UP + RIGHT, buff=0.2)
        )

        self.play(Create(dot), Create(dot_txt))

        # Sampling random points
        self.next_section(skip_animations=False)

        self.play(
            FadeOut(
                img_ref,
                img_ref_rect,
                encoder,
                encoder_txt,
                latent_axes_txt,
                latent_axes_txt_rect,
                z,
                z_txt,
                img_recon,
                img_recon_rect,
                dot,
                dot_txt,
                arrowin,
                arrowout,
                arrow_latent,
            ),
            VGroup(decoder, decoder_txt).animate.move_to(ORIGIN),
            latent_axes.animate.move_to(ORIGIN + 3 * LEFT),
            run_time=2,
        )

        title = (
            Text("Sampling random latent vectors", color=WHITE).scale(0.8).to_edge(UP)
        )
        self.play(Write(title))
        self.wait(6)
        self.play(
            Indicate(decoder, scale_factor=1.4, color=decoder.color),
            Indicate(decoder_txt, scale_factor=1.4, color=decoder_txt.color),
            run_time=2,
        )

        dot1 = Dot(latent_axes.c2p(0.1, 0.7), color=BLUE)
        dot1_txt = (
            MathTex(r"z_1", color=BLUE).scale(0.8).next_to(dot1, UP + RIGHT, buff=0.2)
        )
        dot2 = Dot(latent_axes.c2p(0.7, 0.3), color=BLUE)
        dot2_txt = (
            MathTex(r"z_2", color=BLUE).scale(0.8).next_to(dot2, UP + RIGHT, buff=0.2)
        )
        dot3 = Dot(latent_axes.c2p(0.5, 0.5), color=BLUE)
        dot3_txt = (
            MathTex(r"z_3", color=BLUE).scale(0.8).next_to(dot3, UP + RIGHT, buff=0.2)
        )
        dot4 = Dot(latent_axes.c2p(0.3, 0.1), color=BLUE)
        dot_4_txt = (
            MathTex(r"z_4", color=BLUE).scale(0.8).next_to(dot4, UP + RIGHT, buff=0.2)
        )

        rand_img1 = (
            ImageMobject("images/randoms/reconstructed_0.png")
            .set_resampling_algorithm(RESAMPLING_ALGORITHMS["none"])
            .scale(8)
            .next_to(VGroup(decoder, decoder_txt), RIGHT, buff=1.5)
        )
        rand_img1_rect = SurroundingRectangle(rand_img1, color=BLUE, buff=0.1)

        arrowout = Arrow(
            VGroup(decoder, decoder_txt).get_right(),
            rand_img1.get_left(),
            color=BLUE,
        )

        self.play(Create(dot1), Create(dot1_txt))
        self.play(
            LaggedStart(
                GrowArrow(arrowout),
                AnimationGroup(FadeIn(rand_img1_rect), FadeIn(rand_img1)),
                lag_ratio=0.5,
            )
        )

        rand_img2 = (
            ImageMobject("images/randoms/reconstructed_1.png")
            .set_resampling_algorithm(RESAMPLING_ALGORITHMS["none"])
            .scale(8)
            .next_to(VGroup(decoder, decoder_txt), RIGHT, buff=1.5)
        )
        rand_img2_rect = SurroundingRectangle(rand_img2, color=BLUE, buff=0.1)

        self.play(
            Create(dot2),
            Create(dot2_txt),
            Group(rand_img1, rand_img1_rect).animate.shift(3 * UP + 2 * RIGHT),
            FadeOut(arrowout),
        )
        self.play(
            LaggedStart(
                GrowArrow(arrowout),
                AnimationGroup(FadeIn(rand_img2_rect), FadeIn(rand_img2)),
                lag_ratio=0.5,
            )
        )

        rand_img3 = (
            ImageMobject("images/randoms/reconstructed_2.png")
            .set_resampling_algorithm(RESAMPLING_ALGORITHMS["none"])
            .scale(8)
            .next_to(VGroup(decoder, decoder_txt), RIGHT, buff=1.5)
        )
        rand_img3_rect = SurroundingRectangle(rand_img3, color=BLUE, buff=0.1)

        self.play(
            Create(dot3),
            Create(dot3_txt),
            Group(rand_img2, rand_img2_rect).animate.shift(UP + 2 * RIGHT),
            FadeOut(arrowout),
        )
        self.play(
            LaggedStart(
                GrowArrow(arrowout),
                AnimationGroup(FadeIn(rand_img3_rect), FadeIn(rand_img3)),
                lag_ratio=0.5,
            )
        )

        rand_img4 = (
            ImageMobject("images/randoms/reconstructed_3.png")
            .set_resampling_algorithm(RESAMPLING_ALGORITHMS["none"])
            .scale(8)
            .next_to(VGroup(decoder, decoder_txt), RIGHT, buff=1.5)
        )

        rand_img4_rect = SurroundingRectangle(rand_img4, color=BLUE, buff=0.1)

        self.play(
            Create(dot4),
            Create(dot_4_txt),
            Group(rand_img3, rand_img3_rect).animate.shift(DOWN + 2 * RIGHT),
            FadeOut(arrowout),
        )

        self.play(
            LaggedStart(
                GrowArrow(arrowout),
                AnimationGroup(FadeIn(rand_img4_rect), FadeIn(rand_img4)),
                lag_ratio=0.5,
            )
        )

        self.play(
            Group(rand_img4, rand_img4_rect).animate.shift(3 * DOWN + 2 * RIGHT),
            FadeOut(arrowout),
        )

        self.play(
            FadeOut(
                rand_img1,
                rand_img1_rect,
                dot1,
                dot1_txt,
                rand_img2,
                rand_img2_rect,
                dot2,
                dot2_txt,
                rand_img3,
                rand_img3_rect,
                dot3,
                dot3_txt,
                rand_img4,
                rand_img4_rect,
                dot4,
                dot_4_txt,
                title,
            )
        )

        # Sampling points near another point
        self.next_section(skip_animations=False)

        self.play(VGroup(latent_axes, decoder, decoder_txt).animate.shift(3 * RIGHT))

        VGroup(encoder, encoder_txt).move_to(8 * LEFT)
        self.add(encoder, encoder_txt)

        self.play(
            VGroup(encoder, encoder_txt).animate.next_to(latent_axes, LEFT, buff=0.5)
        )

        title = (
            Text("Sampling latent vectors near a reference", color=WHITE)
            .scale(0.6)
            .to_edge(UP)
        )
        self.play(Write(title))

        img_ref = (
            ImageMobject("images/near/original.png")
            .set_resampling_algorithm(RESAMPLING_ALGORITHMS["none"])
            .scale(8)
            .next_to(encoder, LEFT, buff=1)
        )

        img_ref_rect = SurroundingRectangle(img_ref, color=RED, buff=0.1)

        arrowin = Arrow(img_ref.get_right(), encoder.get_left(), color=RED)

        self.play(Create(img_ref_rect), FadeIn(img_ref))
        self.play(GrowArrow(arrowin))

        dot_ref = Dot(latent_axes.c2p(0.5, 0.5), color=RED)
        dot_ref_txt = (
            MathTex(r"z_{\text{ref}}", color=RED)
            .scale(0.8)
            .next_to(dot_ref, DOWN, buff=0.2)
        )
        self.play(Create(dot_ref), Write(dot_ref_txt))

        dot_near_1 = Dot(latent_axes.c2p(0.6, 0.6), color=BLUE)
        dot_near_1_txt = (
            MathTex(r"z_1", color=BLUE).scale(0.8).next_to(dot_near_1, UP, buff=0.2)
        )
        img_near_1 = (
            ImageMobject("images/near/recon_0.png")
            .set_resampling_algorithm(RESAMPLING_ALGORITHMS["none"])
            .scale(8)
            .next_to(VGroup(decoder, decoder_txt), RIGHT, buff=1)
        )
        img_near_1_rect = SurroundingRectangle(img_near_1, color=BLUE, buff=0.1)

        arrowout = Arrow(
            VGroup(decoder, decoder_txt).get_right(),
            img_near_1.get_left(),
            color=BLUE,
        )

        self.play(Create(dot_near_1), Write(dot_near_1_txt))
        self.play(
            LaggedStart(
                GrowArrow(arrowout),
                AnimationGroup(FadeIn(img_near_1_rect), FadeIn(img_near_1)),
                lag_ratio=0.5,
            )
        )

        dot_near_2 = Dot(latent_axes.c2p(0.35, 0.4), color=BLUE)
        dot_near_2_txt = (
            MathTex(r"z_2", color=BLUE).scale(0.8).next_to(dot_near_2, UP, buff=0.2)
        )

        img_near_2 = (
            ImageMobject("images/near/recon_1.png")
            .set_resampling_algorithm(RESAMPLING_ALGORITHMS["none"])
            .scale(8)
            .next_to(VGroup(decoder, decoder_txt), RIGHT, buff=1)
        )

        img_near_2_rect = SurroundingRectangle(img_near_2, color=BLUE, buff=0.1)

        self.play(
            Create(dot_near_2),
            Write(dot_near_2_txt),
            FadeOut(arrowout),
            Group(img_near_1, img_near_1_rect).animate.shift(2.5 * UP),
        )

        self.play(
            LaggedStart(
                GrowArrow(arrowout),
                AnimationGroup(FadeIn(img_near_2_rect), FadeIn(img_near_2)),
                lag_ratio=0.5,
            )
        )

        dot_near_3 = Dot(latent_axes.c2p(0.5, 0.7), color=BLUE)
        dot_near_3_txt = (
            MathTex(r"z_3", color=BLUE).scale(0.8).next_to(dot_near_3, UP, buff=0.2)
        )

        img_near_3 = (
            ImageMobject("images/near/recon_2.png")
            .set_resampling_algorithm(RESAMPLING_ALGORITHMS["none"])
            .scale(8)
            .next_to(VGroup(decoder, decoder_txt), RIGHT, buff=1)
        )

        img_near_3_rect = SurroundingRectangle(img_near_3, color=BLUE, buff=0.1)

        self.play(
            Create(dot_near_3),
            Write(dot_near_3_txt),
            FadeOut(arrowout),
            Group(img_near_2, img_near_2_rect).animate.shift(2.5 * DOWN),
        )

        self.play(
            LaggedStart(
                GrowArrow(arrowout),
                AnimationGroup(FadeIn(img_near_3_rect), FadeIn(img_near_3)),
                lag_ratio=0.5,
            )
        )

        solution_txt = Text("Organize the latent space !", color=WHITE).scale(1.2)

        self.play(
            FadeOut(
                encoder,
                encoder_txt,
                img_ref,
                img_ref_rect,
                arrowin,
                latent_axes,
                decoder,
                decoder_txt,
                arrowout,
                dot_ref,
                dot_ref_txt,
                dot_near_1,
                dot_near_1_txt,
                dot_near_2,
                dot_near_2_txt,
                dot_near_3,
                dot_near_3_txt,
                img_near_1,
                img_near_1_rect,
                img_near_2,
                img_near_2_rect,
                img_near_3,
                img_near_3_rect,
                title,
            ),
            run_time=2,
        )
        self.play(Write(solution_txt), run_time=2)

        self.play(FadeOut(solution_txt))

        self.wait(2)


if __name__ == "__main__":
    scene = Scene1_2()
    scene.render()
