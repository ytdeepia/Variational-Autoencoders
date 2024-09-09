from manim import *
from manim_voiceover import VoiceoverScene


class Bonus(VoiceoverScene):
    def construct(self):

        self.wait(2)

        approx = MathTex(r"q(z|x)", r"\approx", r"p(z|x)")
        logpx = MathTex(r"{{\log p(x)}}")
        marginalize = MathTex(r"{{\log p(x) = \log \int p(x, z) dz}}")
        trick = MathTex(r"{{\log p(x) = \log \int \frac{q(z|x)}{q(z|x)} p(x, z) dz}}")
        expectation = MathTex(
            r"{{\log p(x) = \log \mathbb{E}_{q(z|x)} \left[ \frac{p(x, z)}{q(z|x)} \right]}}"
        )
        inequality = MathTex(
            r"\log p(x)",
            r"\geq",
            r"\mathbb{E}_{q(z|x)} \left[ \log \frac{p(x, z)}{q(z|x)} \right]",
        )
        inequality_elbo = MathTex(r"{{\log p(x) \geq \mathcal{L}(x)}}")
        elbo = MathTex(
            r"{{\mathcal{L}(x) = \mathbb{E}_{q(z|x)} \left[ \log \frac{p(x, z)}{q(z|x)} \right]}}"
        )
        bayes = MathTex(r"p(x, z) = p(x|z)p(z)")
        elbo_bayes = MathTex(
            r"\mathcal{L}(x) =",
            r"\mathbb{E}_{q(z|x)} \left[ \log p(x|z) +",
            r"\log \frac{p(z)}{q(z|x)} \right]",
        )
        elbo_final = MathTex(
            r"\mathcal{L}(x) =",
            r"\mathbb{E}_{q(z|x)} \left[ \log p(x|z) \right]",
            r"-",
            r"\mathrm{KL}(q(z|x)~|~p(z))",
        )

        # Deriving the ELBO inequality
        self.next_section(skip_animations=False)

        self.play(Write(approx))
        self.play(
            ShowPassingFlash(
                SurroundingRectangle(approx[0], color=WHITE, buff=0.1),
                time_width=0.4,
            ),
            run_time=1.5,
        )

        self.play(FadeOut(approx))
        self.play(Write(logpx))
        self.wait(3)
        self.play(Indicate(logpx, scale_factor=1.4, color=WHITE))

        marginalize.to_corner(UL)
        self.play(Transform(logpx, marginalize))

        trick = trick.next_to(logpx, DOWN, buff=0.5, aligned_edge=LEFT)
        expectation = expectation.next_to(trick, DOWN, buff=0.5, aligned_edge=LEFT)
        inequality = inequality.next_to(expectation, DOWN, buff=0.5, aligned_edge=LEFT)

        marginalize_txt = Tex("Marginalize").next_to(marginalize, RIGHT, buff=2)
        expectation_txt = Tex("Expectation formula").next_to(expectation, RIGHT, buff=2)
        inequality_txt = Tex("Jensen's inequality").to_edge(RIGHT, buff=2)
        jensen_inequality = MathTex(r"\mathbb{E}[f(x)] \geq f(\mathbb{E}[x])").next_to(
            inequality_txt, DOWN, buff=0.5
        )

        self.play(FadeIn(marginalize_txt))

        self.play(FadeOut(marginalize_txt), run_time=0.9)

        self.play(Write(trick))

        self.play(Write(expectation))
        self.play(FadeIn(expectation_txt))

        self.play(FadeOut(expectation_txt), run_time=0.8)

        rect = SurroundingRectangle(jensen_inequality, buff=0.2, color=RED)
        self.play(FadeIn(inequality_txt, jensen_inequality), Create(rect))
        self.wait(2)
        self.play(Write(inequality))

        self.play(FadeOut(inequality_txt, jensen_inequality, rect), run_time=1)

        self.play(
            FadeOut(logpx, trick, expectation), inequality.animate.move_to(ORIGIN)
        )

        self.wait(1)

        rect = SurroundingRectangle(inequality[2], buff=0.2, color=BLUE)
        self.play(Create(rect), run_time=1.5)
        elbo_txt = Tex("Evidence Lower Bound").next_to(rect, UP, buff=0.5)
        self.play(Write(elbo_txt))

        rect2 = SurroundingRectangle(inequality[0], buff=0.2, color=BLUE)
        self.play(Transform(rect, rect2), run_time=1.5)
        evidence_txt = Tex("Evidence").next_to(rect, UP, buff=0.5)
        self.play(Write(evidence_txt))

        self.play(FadeOut(rect, evidence_txt, elbo_txt), run_time=1)

        self.play(Transform(inequality, inequality_elbo))
        elbo.next_to(inequality, DOWN, buff=0.5)
        self.play(Write(elbo))

        self.play(
            ShowPassingFlash(
                SurroundingRectangle(elbo, color=WHITE, buff=0.1), time_width=0.6
            )
        )

        self.play(FadeOut(inequality), elbo.animate.move_to(ORIGIN))

        # Developing the ELBO
        self.next_section(skip_animations=False)

        bayes.next_to(elbo, UP, buff=1)
        self.play(Write(bayes))

        self.play(FadeOut(bayes))
        self.play(Transform(elbo, elbo_bayes))

        self.play(
            ShowPassingFlash(
                SurroundingRectangle(elbo_bayes[2], buff=0.2, color=WHITE)
            ),
            run_time=2,
        )
        self.wait(1)
        elbo_final.next_to(elbo, DOWN, buff=1)
        self.play(Write(elbo_final))

        self.play(
            ShowPassingFlash(
                SurroundingRectangle(elbo_final[3], buff=0.2, color=WHITE)
            ),
            run_time=2,
        )
        elbo_final_copy = elbo_final.copy()
        elbo_final_copy.move_to(elbo)

        self.play(FadeOut(elbo_final), Transform(elbo, elbo_final_copy))

        self.play(FadeOut(elbo), run_time=1)

        self.wait(2)


if __name__ == "__main__":
    scene = Bonus()
    scene.render()
