from manim import *
from manim_voiceover import VoiceoverScene
import numpy as np
from scipy.stats import norm
from sklearn.decomposition import PCA


class Scene3_2(VoiceoverScene):
    def construct(self):
        self.wait(2)

        num_steps = 64
        img = (
            ImageMobject(f"./interpolations/interpolated_{1}.png")
            .set_resampling_algorithm(RESAMPLING_ALGORITHMS["none"])
            .scale(8)
        )

        # Show the latent space
        self.next_section(skip_animations=False)

        latent_codes = np.load("latents/latent_codes_final.npy")
        labels = np.load("latents/labels.npy")
        pca = PCA(n_components=2)
        pca.fit(latent_codes)

        pca_codes = pca.transform(latent_codes)

        latent_axes = Axes(
            x_range=[-5, 5, 1],
            y_range=[-5, 5, 1],
            x_length=6,
            y_length=6,
            axis_config={
                "color": WHITE,
                "include_tip": True,
                "include_ticks": True,
            },
        ).shift(0.4 * DOWN)
        latent_axes_title = (
            Tex("Latent Space PCA", color=WHITE)
            .scale(0.8)
            .next_to(latent_axes, UP, buff=0.5)
        )
        latent_axes_title_ul = Underline(latent_axes_title)

        colors = [
            RED,
            GREEN,
            BLUE,
            YELLOW,
            PURPLE,
            ORANGE,
            PINK,
            TEAL,
            DARK_BROWN,
            GREY,
        ]

        latent_points = VGroup(
            *[
                Dot(
                    point=latent_axes.c2p(code[0], code[1]),
                    color=colors[labels[i]],
                    radius=0.05,
                    fill_opacity=0.6,
                )
                for i, code in enumerate(pca_codes)
            ]
        )

        self.play(
            Create(latent_axes),
            LaggedStart(
                Write(latent_axes_title),
                GrowFromEdge(latent_axes_title_ul, LEFT),
                lag_ratio=0.7,
            ),
        )
        self.play(LaggedStartMap(Create, latent_points, run_time=4))

        self.wait(0.7)

        plot_latent = VGroup(
            latent_axes, latent_points, latent_axes_title, latent_axes_title_ul
        )
        self.play(plot_latent.animate.scale(0.6).to_edge(LEFT, buff=1.5))

        # Show some reconstructions
        self.next_section(skip_animations=False)

        title_recons = (
            Tex("Reconstructions", color=WHITE).scale(0.8).to_corner(UR, buff=0.5)
        )
        title_ref = (
            Tex("Reference", color=WHITE)
            .scale(0.8)
            .next_to(title_recons, LEFT, buff=0.5)
        )

        self.play(
            LaggedStart(Write(title_ref), Write(title_recons), lag_ratio=0.5),
            run_time=2,
        )

        recons = Group()
        refs = Group()

        for i in range(0, 5):
            recon = (
                ImageMobject(f"./reconstructions/reconstruction_{i}.png")
                .set_resampling_algorithm(RESAMPLING_ALGORITHMS["none"])
                .scale(4)
            )
            recon.add(SurroundingRectangle(recon, buff=0.1, color=WHITE))
            recons.add(recon)
            ref = (
                ImageMobject(f"./reconstructions/ref_{i}.png")
                .set_resampling_algorithm(RESAMPLING_ALGORITHMS["none"])
                .scale(4)
            )
            ref.add(SurroundingRectangle(ref, buff=0.1, color=WHITE))

            refs.add(ref)
        recons.arrange(DOWN, buff=0.2).next_to(title_recons, DOWN, buff=0.5)
        refs.arrange(DOWN, buff=0.2).next_to(title_ref, DOWN, buff=0.5)

        for i in range(0, 5):
            self.play(
                FadeIn(refs[i]),
                GrowFromPoint(recons[i], latent_axes.get_center()),
                run_time=1,
            )

        self.play(FadeOut(recons, refs, title_recons, title_ref), run_time=1)

        # Random samples
        self.next_section(skip_animations=False)

        title_samples = (
            Tex("Random Samples", color=WHITE)
            .scale(0.8)
            .to_edge(RIGHT, buff=1.5)
            .to_edge(UP, buff=0.5)
        )

        self.play(Write(title_samples), run_time=1)

        samples = Group()
        for i in range(0, 5):
            sample = (
                ImageMobject(f"./samples/sample_{i}.png")
                .set_resampling_algorithm(RESAMPLING_ALGORITHMS["none"])
                .scale(4)
            )

            sample.add(SurroundingRectangle(sample, buff=0.1, color=WHITE))
            samples.add(sample)

        samples.arrange(DOWN, buff=0.2).next_to(title_samples, DOWN, buff=0.5)

        for i in range(0, 5):
            self.play(GrowFromPoint(samples[i], latent_axes.get_center()), run_time=1)

        self.play(FadeOut(samples, title_samples), run_time=1)

        # Samples close to a given image
        self.next_section(skip_animations=False)

        self.play(
            plot_latent.animate.scale(1.3).move_to(ORIGIN),
            FadeOut(latent_points),
            run_time=1,
        )

        z_dot = Dot(
            point=latent_axes.c2p(0.2, 0.3),
            color=PURPLE,
            radius=0.1,
            fill_opacity=0.7,
        )

        self.play(Create(z_dot), run_time=1)

        ref = (
            ImageMobject(f"./samples_near/origin.png")
            .set_resampling_algorithm(RESAMPLING_ALGORITHMS["none"])
            .scale(2)
            .move_to(z_dot)
        )

        self.play(FadeOut(z_dot), FadeIn(ref), run_time=1)

        near_1 = (
            ImageMobject(f"./samples_near/near_0.png")
            .set_resampling_algorithm(RESAMPLING_ALGORITHMS["none"])
            .scale(2)
            .next_to(z_dot, UP, buff=0.25)
        )
        near_2 = (
            ImageMobject(f"./samples_near/near_1.png")
            .set_resampling_algorithm(RESAMPLING_ALGORITHMS["none"])
            .scale(2)
            .next_to(z_dot, RIGHT, buff=0.25)
        )
        near_3 = (
            ImageMobject(f"./samples_near/near_2.png")
            .set_resampling_algorithm(RESAMPLING_ALGORITHMS["none"])
            .scale(2)
            .next_to(z_dot, DOWN, buff=0.25)
        )
        near_4 = (
            ImageMobject(f"./samples_near/near_3.png")
            .set_resampling_algorithm(RESAMPLING_ALGORITHMS["none"])
            .scale(2)
            .next_to(z_dot, LEFT, buff=0.25)
        )

        self.play(
            LaggedStart(
                GrowFromPoint(near_1, z_dot.get_center()),
                GrowFromPoint(near_2, z_dot.get_center()),
                GrowFromPoint(near_3, z_dot.get_center()),
                GrowFromPoint(near_4, z_dot.get_center()),
                lag_ratio=0.5,
            ),
            run_time=4,
        )

        mid_1 = (
            ImageMobject(f"./samples_near/mid_0.png")
            .set_resampling_algorithm(RESAMPLING_ALGORITHMS["none"])
            .scale(2)
            .move_to(z_dot)
            .shift(0.75 * (UP + RIGHT))
        )
        mid_2 = (
            ImageMobject(f"./samples_near/mid_1.png")
            .set_resampling_algorithm(RESAMPLING_ALGORITHMS["none"])
            .scale(2)
            .move_to(z_dot)
            .shift(0.75 * (DOWN + RIGHT))
        )
        mid_3 = (
            ImageMobject(f"./samples_near/mid_2.png")
            .set_resampling_algorithm(RESAMPLING_ALGORITHMS["none"])
            .scale(2)
            .move_to(z_dot)
            .shift(0.75 * (DOWN + LEFT))
        )
        mid_4 = (
            ImageMobject(f"./samples_near/mid_3.png")
            .set_resampling_algorithm(RESAMPLING_ALGORITHMS["none"])
            .scale(2)
            .move_to(z_dot)
            .shift(0.75 * (UP + LEFT))
        )

        self.play(
            LaggedStart(
                GrowFromPoint(mid_1, z_dot.get_center()),
                GrowFromPoint(mid_2, z_dot.get_center()),
                GrowFromPoint(mid_3, z_dot.get_center()),
                GrowFromPoint(mid_4, z_dot.get_center()),
                lag_ratio=0.5,
            ),
            run_time=4,
        )

        far_1 = (
            ImageMobject(f"./samples_near/far_0.png")
            .set_resampling_algorithm(RESAMPLING_ALGORITHMS["none"])
            .scale(2)
            .next_to(near_1, UP, buff=0.25)
        )
        far_2 = (
            ImageMobject(f"./samples_near/far_1.png")
            .set_resampling_algorithm(RESAMPLING_ALGORITHMS["none"])
            .scale(2)
            .next_to(near_2, RIGHT, buff=0.25)
        )
        far_3 = (
            ImageMobject(f"./samples_near/far_2.png")
            .set_resampling_algorithm(RESAMPLING_ALGORITHMS["none"])
            .scale(2)
            .next_to(near_3, DOWN, buff=0.25)
        )
        far_4 = (
            ImageMobject(f"./samples_near/far_3.png")
            .set_resampling_algorithm(RESAMPLING_ALGORITHMS["none"])
            .scale(2)
            .next_to(near_4, LEFT, buff=0.25)
        )

        self.play(
            LaggedStart(
                GrowFromPoint(far_1, z_dot.get_center()),
                GrowFromPoint(far_2, z_dot.get_center()),
                GrowFromPoint(far_3, z_dot.get_center()),
                GrowFromPoint(far_4, z_dot.get_center()),
                lag_ratio=0.5,
            ),
            run_time=4,
        )

        self.play(
            FadeOut(
                ref,
                near_1,
                near_2,
                near_3,
                near_4,
                mid_1,
                mid_2,
                mid_3,
                mid_4,
                far_1,
                far_2,
                far_3,
                far_4,
            ),
            run_time=0.8,
        )

        # Interpolations
        self.next_section(skip_animations=False)

        self.play(
            VGroup(latent_axes, latent_axes_title, latent_axes_title_ul)
            .animate.scale(0.8)
            .to_edge(LEFT, buff=1.5),
            run_time=1,
        )

        point1 = Dot(
            point=latent_axes.c2p(-2, -2),
            color=PURPLE,
            radius=0.1,
            fill_opacity=0.7,
        )

        img1 = (
            ImageMobject(f"./interpolations/interpolated_{1}.png")
            .set_resampling_algorithm(RESAMPLING_ALGORITHMS["none"])
            .scale(4)
            .next_to(point1, UP, buff=0.5)
        )

        point2 = Dot(
            point=latent_axes.c2p(2, 2),
            color=ORANGE,
            radius=0.1,
            fill_opacity=0.7,
        )

        img2 = (
            ImageMobject(f"./interpolations/interpolated_{64}.png")
            .set_resampling_algorithm(RESAMPLING_ALGORITHMS["none"])
            .scale(4)
            .next_to(point2, UP, buff=0.5)
        )

        self.play(LaggedStart(Create(point1), FadeIn(img1), lag_ratio=0.5), run_time=1)
        self.play(LaggedStart(Create(point2), FadeIn(img2), lag_ratio=0.5), run_time=1)

        interpolation_title = (
            Tex("Interpolation", color=WHITE)
            .scale(0.8)
            .shift(2 * UP + 0.2 * config.frame_width * RIGHT)
        )

        img = img1.copy().scale(2).next_to(interpolation_title, DOWN, buff=1.5)
        img_rect = SurroundingRectangle(img, buff=0.1, color=WHITE)

        self.play(Write(interpolation_title), FadeIn(img, img_rect), run_time=1)

        point1_coord = np.asarray([-2, -2])
        point2_coord = np.asarray([2, 2])

        coords = [
            ((1 - t) * point1_coord + t * point2_coord).tolist()
            for t in np.linspace(0, 1, 64)
        ]

        trackerpoint = Dot(
            point=latent_axes.c2p(*coords[0]),
            color=WHITE,
            radius=0.05,
            fill_opacity=0.7,
        )

        for i in range(2, 65):
            new_img = (
                ImageMobject(f"./interpolations/interpolated_{i}.png")
                .set_resampling_algorithm(RESAMPLING_ALGORITHMS["none"])
                .scale(8)
                .move_to(img)
            )

            self.remove(img)
            self.add(new_img)
            self.play(
                trackerpoint.animate.move_to(latent_axes.c2p(*coords[i - 2])),
                run_time=0.1,
            )
            img = new_img

        self.play(
            FadeOut(
                trackerpoint,
                img,
                img_rect,
                interpolation_title,
                img1,
                img2,
                point1,
                point2,
            ),
            run_time=1,
        )

        self.play(
            VGroup(latent_axes, latent_axes_title, latent_axes_title_ul)
            .animate.scale(1.3)
            .move_to(ORIGIN),
            run_time=1,
        )

        latent_points = VGroup(
            *[
                Dot(
                    point=latent_axes.c2p(code[0], code[1]),
                    color=colors[labels[i]],
                    radius=0.05,
                    fill_opacity=0.6,
                )
                for i, code in enumerate(pca_codes)
            ]
        )

        self.play(FadeIn(latent_points), run_time=1)

        self.play(
            FadeOut(
                latent_axes, latent_axes_title_ul, latent_axes_title, latent_points
            ),
            run_time=1,
        )

        self.wait(2)


if __name__ == "__main__":
    scene = Scene3_2()
    scene.render()
