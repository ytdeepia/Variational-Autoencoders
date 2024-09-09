from manim import *
from manim_voiceover import VoiceoverScene
import numpy as np
from sklearn.decomposition import PCA


class Scene3_1(VoiceoverScene):
    def construct(self):

        # Recap the training process
        self.next_section(skip_animations=False)

        ref_img = (
            ImageMobject("./images/ref_7.png")
            .set_resampling_algorithm(RESAMPLING_ALGORITHMS["none"])
            .scale(8)
        )
        ref_img_rect = SurroundingRectangle(
            ref_img, buff=0, color=WHITE, stroke_width=2
        )
        ref_img = Group(ref_img, ref_img_rect)
        recons_img = (
            ImageMobject("./images/recons_7.png")
            .set_resampling_algorithm(RESAMPLING_ALGORITHMS["none"])
            .scale(8)
        )
        recons_img_rect = SurroundingRectangle(
            recons_img, buff=0, color=WHITE, stroke_width=2
        )
        recons_img = Group(recons_img, recons_img_rect)

        encoder = Polygon(
            [-1, 1, 0], [1, 0.6, 0], [1, -0.6, 0], [-1, -1, 0], color=PURPLE
        )
        encoder_txt = Tex("Encoder").scale(0.8).move_to(encoder)
        encoder = VGroup(encoder, encoder_txt)

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
            [-1, 0.6, 0], [1, 1, 0], [1, -1, 0], [-1, -0.6, 0], color=PURPLE
        )
        decoder_txt = Tex("Decoder").scale(0.8).move_to(decoder)
        decoder = VGroup(decoder, decoder_txt).next_to(z, RIGHT, buff=0.2)

        vae = VGroup(encoder, params, distrib, sample, z, decoder).move_to(ORIGIN)
        self.play(FadeIn(vae))

        ref_img.next_to(vae, LEFT, buff=0.5)
        ref_img_txt = MathTex(r"x").scale(0.8).next_to(ref_img, DOWN, buff=0.2)
        ref_img = Group(ref_img, ref_img_txt)
        recons_img.next_to(vae, RIGHT, buff=0.5)
        recons_img_txt = MathTex(r"x'").scale(0.8).next_to(recons_img, DOWN, buff=0.2)
        recons_img = Group(recons_img, recons_img_txt)
        self.play(FadeIn(ref_img))
        self.wait(1)
        self.play(ApplyWave(vae), run_time=2)
        self.wait(1)
        self.play(FadeIn(recons_img))

        self.play(Group(vae, ref_img, recons_img).animate.to_edge(UP))

        loss = MathTex(
            r"\mathcal{L} =",
            r"\mathcal{L}_{KL}(\mathcal{N}(\mu, \sigma) ~|~ \mathcal{N}(0, 1))",
            r"+",
            r"\mathcal{L}_{2}(x, x')}",
        )

        self.play(Write(loss))

        rect_kl_loss = SurroundingRectangle(loss[1], buff=0.1)
        rect_recons_loss = SurroundingRectangle(loss[3], color=BLUE, buff=0.1)

        self.play(Create(rect_recons_loss))

        self.play(Indicate(params, scale_factor=1.3, color=YELLOW), run_time=2)
        self.play(Indicate(params, scale_factor=1.3, color=YELLOW), run_time=2)

        self.play(Transform(rect_recons_loss, rect_kl_loss))

        explicit_kl = (
            MathTex(
                r"\mathcal{L}_{KL} = -\frac{1}{2} (1 + \log(\sigma^2) - \mu^2 - \sigma^2)"
            )
            .scale(0.8)
            .next_to(loss, DOWN, buff=0.5)
        )

        self.play(Write(explicit_kl))

        self.play(
            FadeOut(rect_recons_loss, explicit_kl, loss, vae, ref_img, recons_img)
        )

        # Show the training process
        self.next_section(skip_animations=False)

        # Create the axes
        axes_loss = (
            Axes(
                x_range=[0, 530, 50],
                y_range=[-0.1, 0.25, 0.1],
                axis_config={
                    "color": WHITE,
                },
                x_axis_config={"include_numbers": True},
            )
            .scale(0.3)
            .to_corner(UR, buff=0.5)
        )

        axes_loss_title = (
            Tex("Loss Evolution").scale(0.5).next_to(axes_loss, UP, buff=0.2)
        )
        axes_loss_title_ul = Underline(axes_loss_title)

        legend_loss = (
            VGroup(
                MathTex(r"\mathcal{L}_{KL}", color=RED).scale(0.5),
                MathTex(r"\mathcal{L}_{2}", color=BLUE).scale(0.5),
            )
            .arrange(RIGHT, buff=0.5)
            .to_corner(UR, buff=0.5)
        )

        axes_latent_space = Axes(
            x_range=[-5, 5, 1],
            y_range=[-5, 5, 1],
            axis_config={
                "color": WHITE,
                "include_tip": True,
                "include_ticks": True,
            },
        ).scale(0.9)

        axes_latent_space_title = (
            Tex("Latent Space PCA").scale(0.8).next_to(axes_latent_space, UP, buff=0.3)
        )
        axes_latent_space_title_ul = Underline(axes_latent_space_title)

        # Create the colorbar
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

        colorbar = VGroup()
        clabels = VGroup()
        for i, color in enumerate(colors):
            rect = Rectangle(
                width=0.4,
                height=0.3,
                color=color,
                fill_opacity=1,
                stroke_width=0,
            )
            srect = SurroundingRectangle(
                rect,
                color=rgb_to_color(color_to_rgb(color) * 0.3),
                stroke_width=4,
                buff=0,
            )
            label = (
                Tex(f"{i}", color=rgb_to_color(color_to_rgb(color) * 0.3))
                .scale(0.5)
                .move_to(rect)
            )
            colorbar.add(VGroup(rect, srect, label))

        colorbar.arrange(DOWN, buff=0.2)
        # for idx, clabel in enumerate(clabels):
        #     clabel.next_to(colorbar[idx], direction=LEFT, buff=0.3)

        colorbar = (
            VGroup(colorbar, clabels)
            .scale_to_fit_height(axes_latent_space.height)
            .next_to(axes_latent_space, LEFT, buff=0.5)
        )

        epoch_counter = Tex("Gradient Step: 0").scale(0.8).to_corner(UL, buff=0.5)
        self.play(
            FadeIn(colorbar, legend_loss),
            FadeIn(epoch_counter),
            Create(axes_loss),
            Create(axes_latent_space),
            FadeIn(axes_latent_space_title, axes_latent_space_title_ul),
            FadeIn(axes_loss_title, axes_loss_title_ul),
            run_time=2,
        )

        latent = np.load("./latents/latent_codes_530.npy")
        labels = np.load("./latents/labels.npy")

        pca = PCA(n_components=2)
        pca.fit(latent)

        latent = np.load(f"./latents/latent_codes_1.npy")
        latent_pca = pca.transform(latent)
        latent_dots = VGroup()

        for idx, p in enumerate(latent_pca):
            dot = Dot(
                axes_latent_space.c2p(p[0], p[1]),
                color=colors[int(labels[idx])],
                radius=0.05,
                fill_opacity=0.7,
            )
            latent_dots.add(dot)

        self.play(FadeIn(latent_dots), run_time=1)

        # Animate the losses and the latent space
        kl_loss = np.load("./latents/kl_losses_full.npy")
        recon_loss = np.load("./latents/recon_losses_full.npy")
        kl_loss = kl_loss * 0.00025
        losses_graph = VGroup()
        skip = 5

        for idx in range(0, 530, skip):
            if idx + skip < len(kl_loss):
                line_kl = Line(
                    start=axes_loss.c2p(idx, kl_loss[idx]),
                    end=axes_loss.c2p(idx + skip, kl_loss[idx + skip]),
                    color=RED,
                    stroke_width=2,
                )

                line_recons = Line(
                    start=axes_loss.c2p(idx, recon_loss[idx]),
                    end=axes_loss.c2p(idx + skip, recon_loss[idx + skip]),
                    color=BLUE,
                    stroke_width=2,
                )

                losses_graph.add(line_kl)
                latent = np.load(f"./latents/latent_codes_{idx+1}.npy")
                latent_pca = pca.transform(latent)
                latent_dots_next = VGroup()

                latent_dots_next = VGroup(
                    *[
                        Dot(
                            axes_latent_space.c2p(p[0], p[1]),
                            color=colors[int(labels[i])],
                            radius=0.02,
                            fill_opacity=0.7,
                        )
                        for i, p in enumerate(latent_pca)
                    ]
                )

                epoch_counter_next = (
                    Tex(f"Gradient Step: {idx+skip}").scale(0.8).to_corner(UL, buff=0.5)
                )
                self.play(
                    Transform(latent_dots, latent_dots_next),
                    Transform(epoch_counter, epoch_counter_next),
                    Create(line_kl),
                    Create(line_recons),
                    run_time=0.5,
                )

        self.wait(2)


if __name__ == "__main__":
    scene = Scene3_1()
    scene.render()
