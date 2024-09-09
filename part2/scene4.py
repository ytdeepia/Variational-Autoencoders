from manim import *
from manim_voiceover import VoiceoverScene


class Scene2_4(ThreeDScene, VoiceoverScene):
    def construct(self):

        self.wait(2)

        elbo = MathTex(
            r"\mathcal{L}(x) =",
            r"\mathbb{E}_{q(z|x)} \left[ \log p(x|z) \right]",
            r"-",
            r"\mathrm{KL}(q(z|x)~|~p(z))",
        )

        self.play(Write(elbo))

        brace_likelihood = Brace(elbo[1], DOWN, buff=0.1)
        likelihood = Tex("Data consistency").next_to(brace_likelihood, DOWN)

        brace_kl = Brace(elbo[3], UP, buff=0.1)
        kl = Tex("Regularization").next_to(brace_kl, UP)

        self.play(GrowFromCenter(brace_likelihood), Write(likelihood))
        self.wait(1)
        self.play(GrowFromCenter(brace_kl), Write(kl))

        self.play(
            FadeOut(likelihood, brace_likelihood, brace_kl, kl),
        )

        self.play(elbo.animate.to_edge(UP, buff=0.5))

        data_dist = (
            SVGMobject(
                "images/data_dist.svg",
                fill_color=[RED_C, RED_E],
                fill_opacity=1,
                stroke_opacity=0,
            )
            .scale(1.5)
            .to_edge(LEFT, buff=1)
        )
        data_title = (
            Tex("Data Distribution", color=RED_D)
            .scale(0.8)
            .next_to(data_dist, DOWN, buff=0.5)
        )
        data_title_ul = Underline(data_title, color=RED_D)

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
            .scale(0.8)
            .next_to(latent_dist, DOWN, buff=0.5)
        )
        latent_title_ul = Underline(latent_title, color=BLUE_D)

        self.play(
            Create(data_dist),
            Create(latent_dist),
            LaggedStart(
                Write(data_title), GrowFromEdge(data_title_ul, LEFT), lag_ratio=0.6
            ),
            LaggedStart(
                Write(latent_title),
                GrowFromEdge(latent_title_ul, LEFT),
                lag_ratio=0.6,
            ),
            run_time=2,
        )

        # Explaining the likelihood term
        self.next_section(skip_animations=False)

        img_ref = (
            ImageMobject("images/ref_0.png")
            .set_resampling_algorithm(RESAMPLING_ALGORITHMS["none"])
            .scale(6)
        )
        img_recon = (
            ImageMobject("images/recon_0.png")
            .set_resampling_algorithm(RESAMPLING_ALGORITHMS["none"])
            .scale(6)
        )
        Group(img_ref, img_recon).arrange(RIGHT, buff=0.5).move_to(ORIGIN)

        img_recon_rect = SurroundingRectangle(img_recon, buff=0.1, color=WHITE)
        img_ref_rect = SurroundingRectangle(img_ref, buff=0.1, color=WHITE)

        point_latent = Dot(latent_dist.get_center(), color=WHITE)
        point_data = Dot(data_dist.get_center(), color=WHITE)

        posterior_arrow = CurvedArrow(
            LEFT + UP,
            RIGHT + UP,
            angle=-PI / 2,
        )
        posterior = MathTex(r"q(z|x)").next_to(posterior_arrow, UP)

        likelihood_arrow = CurvedArrow(
            DOWN + RIGHT,
            DOWN + LEFT,
            angle=-PI / 2,
        )
        likelihood = MathTex(r"p(x|z)").next_to(likelihood_arrow, DOWN)

        self.play(Create(point_data))
        self.play(GrowFromPoint(Group(img_ref, img_ref_rect), data_dist.get_center()))
        self.play(
            LaggedStart(
                Create(posterior_arrow),
                Write(posterior),
                lag_ratio=0.5,
            ),
            run_time=2,
        )

        self.wait(1)
        rect = SurroundingRectangle(elbo[1], buff=0.2, color=WHITE)
        self.play(ShowPassingFlash(rect, time_width=0.6), run_time=2)
        self.wait(1)
        l2 = MathTex(r"\mathbb{E}_{q(z|x)} \left[ \log p(x|z) \right] = L_2").to_edge(
            DOWN, buff=2
        )
        self.play(Write(l2))

        self.play(FadeOut(l2))
        self.wait(1)
        self.play(Create(point_latent))
        self.play(
            LaggedStart(
                FadeOut(point_latent),
                Create(likelihood_arrow),
                Write(likelihood),
                lag_ratio=0.5,
            ),
            run_time=2,
        )
        self.play(
            GrowFromPoint(Group(img_recon, img_recon_rect), latent_dist.get_center())
        )

        self.play(
            FadeOut(
                data_dist,
                data_title,
                data_title_ul,
                point_data,
                likelihood_arrow,
                likelihood,
                posterior_arrow,
                posterior,
                img_recon,
                img_recon_rect,
                img_ref,
                img_ref_rect,
            )
        )

        # Explain the regularization term
        self.next_section(skip_animations=False)

        self.play(
            latent_dist.animate.move_to(0.5 * DOWN),
            FadeOut(latent_title, latent_title_ul),
        )

        rect = SurroundingRectangle(elbo[3], buff=0.2, color=WHITE)
        self.play(ShowPassingFlash(rect, time_width=0.6), run_time=2)

        posterior.move_to(latent_dist)

        self.play(Write(posterior))
        prior_dist = DashedVMobject(
            Circle(radius=3, color=BLUE_D), num_dashes=40
        ).move_to(latent_dist)

        prior = MathTex(r"p(z)", color=BLUE_D).next_to(prior_dist, RIGHT).shift(UP)

        self.play(Create(prior_dist))
        self.play(Write(prior))

        sigma = 3
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

        self.play(Transform(latent_dist, normal), run_time=3)

        self.play(FadeOut(prior_dist, prior, latent_dist, posterior), run_time=1.5)

        # Wrap up
        self.next_section(skip_animations=False)

        self.play(elbo.animate.move_to(ORIGIN))

        brace_likelihood = Brace(elbo[1], DOWN, buff=0.1)
        likelihood = Tex("L2").next_to(brace_likelihood, DOWN)

        brace_kl = Brace(elbo[3], UP, buff=0.1)
        kl = Tex("Latent space regularization").next_to(brace_kl, UP)

        self.play(GrowFromCenter(brace_likelihood), Write(likelihood))
        self.play(GrowFromCenter(brace_kl), Write(kl))
        self.play(FadeOut(elbo, brace_likelihood, likelihood, brace_kl, kl))

        self.wait(2)


if __name__ == "__main__":
    scene = Scene2_4()
    scene.render()
