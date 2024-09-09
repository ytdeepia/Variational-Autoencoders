from manim import *
from manim_voiceover import VoiceoverScene


class Scene2_5(VoiceoverScene):
    def construct(self):
        self.wait(2)

        encoder = Polygon(
            [-1, 1, 0], [1, 0.6, 0], [1, -0.6, 0], [-1, -1, 0], color=PURPLE
        )
        encoder_txt = Tex("Encoder").scale(0.8).move_to(encoder)
        encoder = VGroup(encoder, encoder_txt)

        decoder = Polygon(
            [-1, 0.6, 0], [1, 1, 0], [1, -1, 0], [-1, -0.6, 0], color=PURPLE
        )
        decoder_txt = Tex("Decoder").scale(0.8).move_to(decoder)
        decoder = VGroup(decoder, decoder_txt)

        z = Rectangle(height=1.2, width=0.45, color=BLUE).next_to(encoder, RIGHT)
        z_txt = MathTex(r"z").scale(0.8).move_to(z)
        z = VGroup(z, z_txt)
        autoencoder = (
            VGroup(encoder, z, decoder).arrange(RIGHT, buff=0.5).move_to(ORIGIN)
        )

        axes_latent = (
            Axes(
                x_range=[0, 1, 1],
                y_range=[0, 1, 1],
                x_length=4,
                y_length=4,
                axis_config={"color": WHITE},
            )
            .scale(0.8)
            .to_edge(UP, buff=0.5)
        )
        axes_latent_title = Tex("Latent Space").next_to(axes_latent, LEFT)
        axes_latent_title_ul = Underline(axes_latent_title)

        # Regular Autoencoder
        self.next_section(skip_animations=False)

        txt = Tex("How do we do in practice ?").scale(1.2)
        self.play(Write(txt))

        self.play(FadeOut(txt))

        self.play(Create(encoder))
        self.play(Create(z))
        self.play(Create(decoder))

        self.play(
            LaggedStart(
                autoencoder.animate.shift(1.5 * DOWN),
                AnimationGroup(
                    Create(axes_latent),
                    Write(axes_latent_title),
                    GrowFromEdge(axes_latent_title_ul, LEFT),
                ),
                lag_ratio=0.5,
            ),
            run_time=2,
        )

        z_latent = Dot(axes_latent.c2p(0.3, 0.5), color=BLUE)
        z_latent_txt = MathTex(r"z").scale(0.8).next_to(z_latent, UP)

        self.play(Create(z_latent), Write(z_latent_txt))

        # Variational Autoencoder
        self.next_section(skip_animations=False)

        sample1 = MathTex(r"\sim").scale(0.8).next_to(z, LEFT, buff=0.2)
        distrib = Rectangle(height=1.2, width=1.8, color=RED).next_to(
            sample1, LEFT, buff=0.2
        )
        distrib_txt = MathTex(r"\mathcal{N}(\mu, \sigma)").scale(0.8).move_to(distrib)
        distrib = VGroup(distrib, distrib_txt)

        mu = Rectangle(height=0.6, width=0.45, color=YELLOW)
        mu_txt = MathTex(r"\mu").scale(0.8).move_to(mu)
        sigma = Rectangle(height=0.6, width=0.45, color=YELLOW)
        sigma_txt = MathTex(r"\sigma").scale(0.8).move_to(sigma)

        mu = VGroup(mu, mu_txt)
        sigma = VGroup(sigma, sigma_txt)

        params = VGroup(mu, sigma).arrange(DOWN, buff=0.1).next_to(distrib, LEFT, 0.5)

        self.play(encoder.animate.next_to(params, LEFT, buff=0.5))
        self.play(Create(params))
        self.play(Create(distrib), Create(sample1))

        radius = 1
        colors = color_gradient([RED, BLACK], 100)
        z_latent_dist = VGroup(
            *[
                Circle(
                    radius=(i + 1) * radius / 100,
                    stroke_width=2 * radius,
                    color=colors[i],
                )
                for i in range(100)
            ]
        )
        z_latent_dist.scale(0.5).move_to(z_latent)
        z_latent_dist_txt = (
            MathTex(r"\mathcal{N(\mu, \sigma)}", color=RED)
            .scale(0.8)
            .next_to(z_latent_dist, DOWN)
        )

        self.play(FadeOut(z_latent, z_latent_txt), Create(z_latent_dist), run_time=2)
        self.play(Write(z_latent_dist_txt))

        self.wait(0.9)

        z_latent_1 = Dot(axes_latent.c2p(0.33, 0.45), color=BLUE)
        z_latent_2 = Dot(axes_latent.c2p(0.26, 0.54), color=BLUE)

        self.play(Create(z_latent_1))
        self.play(Create(z_latent_2))

        self.play(
            FadeOut(
                axes_latent,
                axes_latent_title,
                axes_latent_title_ul,
                z_latent_dist,
                z_latent_dist_txt,
                z_latent_1,
                z_latent_2,
            )
        )

        self.play(
            VGroup(encoder, params, distrib, sample1, z, decoder).animate.move_to(
                ORIGIN
            )
        )

        elbo = MathTex(
            r"\mathcal{L}(x, x') = L_2(x, x') + D_{KL}(\mathcal{N}(\mu, \sigma)~ | ~\mathcal{N}(0, 1))"
        ).to_edge(UP, buff=0.5)

        self.play(Write(elbo))

        self.play(
            FadeOut(elbo),
            VGroup(encoder, params, distrib, sample1, z, decoder).animate.shift(UP),
        )

        # Reparameterization Trick
        self.next_section(skip_animations=False)

        txt = Tex("How to backpropagate through the sampling process?")
        txt.to_edge(UP)

        rect = SurroundingRectangle(VGroup(distrib, sample1, z), buff=0.2, color=WHITE)

        self.play(Write(txt))
        self.play(ShowPassingFlash(rect, time_width=0.5), run_time=1.5)

        trick_txt = Tex("Reparameterization Trick").to_edge(UP)
        trick_ul = Underline(trick_txt)

        self.play(Transform(txt, trick_txt))
        self.play(Create(trick_ul))

        add = Rectangle(height=0.5, width=1.8, color=GREEN).move_to(distrib)
        add_txt = MathTex(r"\mu + \sigma \cdot \epsilon").scale(0.8).move_to(add)
        add = VGroup(add, add_txt)

        self.play(FadeOut(distrib, distrib_txt, sample1))
        self.play(Create(add))

        normal_dist = (
            MathTex(r"\mathcal{N}(0, 1)")
            .scale(0.8)
            .to_edge(DOWN, buff=2.5)
            .shift(3 * LEFT)
        )

        sample = MathTex(r"\sim").scale(0.8).next_to(normal_dist, RIGHT, buff=0.2)
        epsilon = Rectangle(height=0.6, width=0.45, color=RED).next_to(
            sample, RIGHT, buff=0.2
        )
        epsilon_txt = MathTex(r"\epsilon").scale(0.8).move_to(epsilon)

        self.play(
            LaggedStart(
                Create(normal_dist),
                Create(sample),
                AnimationGroup(Create(epsilon), Write(epsilon_txt)),
                lag_ratio=0.5,
            ),
            run_time=2,
        )

        line_eps = Line(
            epsilon.get_right() + 0.1 * RIGHT,
            (add.get_x(), epsilon.get_y(), 0),
        )
        arrow_eps = Arrow(line_eps.get_right(), add.get_bottom() + 0.1 * DOWN, buff=0)
        line_eps.stroke_width = arrow_eps.stroke_width

        self.play(Create(line_eps))
        self.play(Create(arrow_eps))

        vae = VGroup(encoder, params, add, z, decoder)

        self.play(
            FadeOut(epsilon, epsilon_txt, line_eps, arrow_eps, normal_dist, sample, vae)
        )

        # Reparameterization Trick demo
        self.next_section(skip_animations=False)

        axes = Axes(
            x_range=[0, 1, 1],
            y_range=[0, 1, 1],
            x_length=5,
            y_length=5,
            axis_config={"color": WHITE},
        ).to_edge(DOWN, buff=1.5)

        mu = (0.3, 0.7)
        sigma = 0.6

        colors = color_gradient([RED, BLACK], 100)
        dist = VGroup(
            *[
                Circle(
                    radius=(i + 1) * sigma / 100,
                    stroke_width=2 * sigma,
                    color=colors[i],
                )
                for i in range(100)
            ]
        ).move_to(axes.c2p(*mu))

        sigma_normal = 1.1
        colors = color_gradient([WHITE, BLACK], 100)
        dist_normal = VGroup(
            *[
                Circle(
                    radius=(i + 1) * sigma_normal / 100,
                    stroke_width=2 * sigma_normal,
                    color=colors[i],
                )
                for i in range(100)
            ]
        ).move_to(axes.c2p(0.65, 0.5))

        self.play(FadeIn(axes))
        self.play(Create(dist))
        self.play(Create(dist_normal))
        self.wait()

        epsilon_sample = (
            MathTex(r"\epsilon \sim \mathcal{N}(0, 1)", color=BLUE)
            .next_to(axes, RIGHT)
            .shift(2 * UP)
        )
        epsilon_sample_rect = SurroundingRectangle(
            epsilon_sample, buff=0.1, color=WHITE
        )

        dot = Dot(axes.c2p(0.6, 0.45), color=BLUE)

        self.play(
            LaggedStart(
                Write(epsilon_sample), Create(epsilon_sample_rect), lag_ratio=0.5
            )
        )
        self.play(Create(dot))

        mutliply_sample = MathTex("\sigma \cdot \epsilon").next_to(
            epsilon_sample, DOWN, buff=1
        )
        mutliply_sample_rect = SurroundingRectangle(
            mutliply_sample, buff=0.1, color=WHITE
        )

        self.play(
            LaggedStart(
                Write(mutliply_sample), Create(mutliply_sample_rect), lag_ratio=0.5
            )
        )
        self.play(VGroup(dist_normal, dot).animate.scale_to_fit_width(dist.width))

        shift_sample = MathTex("\mu + \sigma \cdot \epsilon").next_to(
            mutliply_sample, DOWN, buff=1
        )
        shift_sample_rect = SurroundingRectangle(shift_sample, buff=0.1, color=WHITE)
        self.play(
            LaggedStart(Write(shift_sample), Create(shift_sample_rect), lag_ratio=0.5)
        )
        self.play(VGroup(dist_normal, dot).animate.move_to(dist))

        self.play(FadeOut(dist_normal))

        self.play(
            FadeOut(
                axes,
                dist,
                dot,
                epsilon_sample,
                mutliply_sample,
                shift_sample,
                epsilon_sample_rect,
                mutliply_sample_rect,
                shift_sample_rect,
            )
        )

        # Ending
        self.next_section(skip_animations=False)

        self.play(
            FadeIn(epsilon, epsilon_txt, line_eps, arrow_eps, normal_dist, sample, vae)
        )

        arrow_backprop = Arrow(
            vae.get_right() + 2 * UP,
            vae.get_left() + 2 * UP,
            buff=0,
        )

        backprop_txt = Tex("Backpropagation").next_to(arrow_backprop, UP)
        self.play(FadeOut(txt, trick_ul))
        self.play(
            Create(arrow_backprop),
            run_time=2,
        )
        self.play(Write(backprop_txt))

        self.play(
            FadeOut(
                arrow_backprop,
                backprop_txt,
                epsilon,
                epsilon_txt,
                line_eps,
                arrow_eps,
                normal_dist,
                sample,
                vae,
            )
        )

        self.wait(2)


if __name__ == "__main__":
    scene = Scene2_5()
    scene.render()
