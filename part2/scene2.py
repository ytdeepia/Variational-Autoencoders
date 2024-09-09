from manim import *
from manim_voiceover import VoiceoverScene
import numpy as np
from scipy.stats import norm


class Scene2_2(ThreeDScene, VoiceoverScene):
    def construct(self):

        self.wait(2)

        mean_y = 3
        x_range_y = np.linspace(0, 10, 20)  # 20 points between 0 and 10
        pdf_values_y = norm.pdf(x_range_y, mean_y, 1)
        pdf_values_y /= pdf_values_y.sum()

        bar_colors_y = [
            interpolate_color(BLUE_E, BLUE_A, alpha)
            for alpha in np.linspace(0, 1, len(pdf_values_y))
        ]

        # Create a bar chart with y-axis and value numbers hidden
        bar_chart_y = BarChart(
            values=pdf_values_y,
            y_range=[0, pdf_values_y.max() + 0.01, 0.01],
            x_length=10,
            y_length=4,
            bar_colors=bar_colors_y,
            bar_fill_opacity=0.75,
            bar_stroke_width=1,
            bar_width=1.0,
            y_axis_config={
                "include_numbers": False,
                "include_ticks": False,
                "stroke_opacity": 0,
            },
            x_axis_config={
                "include_numbers": False,
                "include_ticks": False,
                "stroke_opacity": 0,
            },  # Hide x-axis
        )

        bar_chart_y.rotate(90 * DEGREES).to_edge(LEFT)

        mean_x = 7
        x_range_x = np.linspace(0, 10, 20)  # 20 points between 0 and 10
        pdf_values_x = norm.pdf(x_range_x, mean_x, 1)
        pdf_values_x /= pdf_values_x.sum()

        bar_colors_x = [
            interpolate_color(RED_E, RED_A, alpha)
            for alpha in np.linspace(0, 1, len(pdf_values_x))
        ]

        bar_chart_x = BarChart(
            values=pdf_values_x,
            y_range=[0, pdf_values_x.max() + 0.01, 0.01],
            x_length=10,
            y_length=4,
            bar_colors=bar_colors_x,
            bar_fill_opacity=0.75,
            bar_stroke_width=1,
            bar_width=1.0,
            y_axis_config={
                "include_numbers": False,
                "include_ticks": False,
                "stroke_opacity": 0,
            },  # Hide y-axis
            x_axis_config={
                "include_numbers": False,
                "include_ticks": False,
                "stroke_opacity": 0,
            },
        )

        # Intro
        self.next_section(skip_animations=False)

        bar_chart_x.shift(UP)

        txt = Tex("Introduction to Bayesian statistics").to_edge(UP)
        txt_ul = Underline(txt)

        self.play(Write(txt), GrowFromEdge(txt_ul, LEFT))

        X = MathTex(r"X", color=RED).scale(1.5)
        subtitle = Tex("Random Variable").next_to(X, DOWN, buff=0.5)
        Xin = MathTex(r"X", r"\in [0, 10]").scale(1.5)
        Xin[0].set_color(RED)

        self.play(Write(subtitle), FadeIn(X))
        self.wait(1)
        self.play(Transform(X, Xin))

        self.wait(0.9)

        subtitle2 = Tex("Sampling").move_to(subtitle)
        sampling = MathTex(r"x \sim", r"X").scale(1.5).move_to(X)
        sampling[1].set_color(RED)
        self.play(Transform(X, sampling), FadeOut(subtitle))
        self.wait(1)
        self.play(Write(subtitle2))
        self.play(ShowPassingFlash(Underline(subtitle2), time_width=0.5))

        self.wait(0.8)

        self.play(FadeOut(X, subtitle2))
        pX = MathTex("p(x)", color=RED).scale(1.5).to_edge(LEFT, buff=1)
        self.play(Create(bar_chart_x), FadeOut(txt, txt_ul))
        self.play(Write(pX))

        title = Tex("Probability Density Function").next_to(bar_chart_x, DOWN)
        self.play(Write(title))
        self.play(ShowPassingFlash(Underline(title), time_width=0.5))

        rect = SurroundingRectangle(pX, buff=0.3, color=WHITE)
        self.play(ShowPassingFlash(rect, time_width=0.5), run_time=2)

        # Expectation
        line = DashedLine(
            bar_chart_x.c2p(14, 0) + 0.2 * DOWN,
            bar_chart_x.c2p(14, pdf_values_x.max()) + 0.2 * UP,
            color=WHITE,
        )

        self.play(FadeOut(title))
        expectation = MathTex(r"\mathbb{E}[X]").next_to(line, DOWN, buff=0.5)
        subtitle = Tex("Expectation").next_to(expectation, DOWN, buff=0.5)
        self.play(Create(line))
        self.play(Write(expectation))
        self.play(Write(subtitle))

        self.wait(0.8)

        expectation_dev = MathTex(
            r"\mathbb{E}[X] = \int_{-\infty}^{\infty} x", r"p(x)", r"dx"
        ).move_to(expectation)

        expectation_dev[1].set_color(RED)
        self.play(Transform(expectation, expectation_dev))

        # Display the joint distribution
        self.next_section(skip_animations=False)

        self.play(FadeOut(expectation, line, pX, bar_chart_x, subtitle))

        bar_chart_y.next_to(bar_chart_x.axes, LEFT, buff=0)
        bar_chart_x.shift(UP * (bar_chart_y.height / 2 + bar_chart_x.height / 2))

        bars = VGroup(bar_chart_y, bar_chart_x)
        bars.move_to(ORIGIN)

        colors = ["#003f5c", "#58508d", "#bc5090", "#ff6361", "#ffa600"]

        def interpolate_gradient(val):
            """
            Interpolate the color based on the value from the gradient.
            """
            if val <= 0:
                return colors[0]
            elif val >= 1:
                return colors[-1]
            else:
                index = int(val * (len(colors) - 1))
                remainder = val * (len(colors) - 1) - index
                return interpolate_color(
                    ManimColor(colors[index]),
                    ManimColor(colors[index + 1]),
                    remainder,
                )

        # Create lattice
        cell_width = 0.5
        lattice_cells = VGroup()

        for j in range(len(pdf_values_y)):
            for i in range(len(pdf_values_x)):
                # Calculate the product of the probabilities
                cell_value = pdf_values_y[-j] * pdf_values_x[i]
                # Normalize the value for color interpolation
                normalized_value = cell_value / (
                    pdf_values_y.max() * pdf_values_x.max()
                )
                cell_color = interpolate_gradient(normalized_value)
                cell = Prism(
                    dimensions=[cell_width, cell_width, 70 * cell_value],
                    fill_color=cell_color,
                    fill_opacity=0.8,
                    stroke_width=0.5,
                )
                cell.move_to(
                    bar_chart_x.get_corner(DL)
                    + (0.5 * cell_width) * DOWN
                    + (0.5 * cell_width) * RIGHT
                    + (cell_width * i * RIGHT)
                    + (cell_width * j * DOWN)
                    + cell.dimensions[2] / 2 * OUT
                )
                lattice_cells.add(cell)

        graph = VGroup(bars, lattice_cells)
        graph.move_to(ORIGIN + 0.5 * (LEFT + UP)).scale(0.5)

        self.play(Create(bar_chart_x), run_time=1)
        self.play(Create(bar_chart_y), run_time=1)

        self.wait(0.6)

        # Display the joint distribution

        self.play(FadeIn(lattice_cells))
        self.move_camera(phi=60 * DEGREES, theta=-45 * DEGREES, run_time=2)
        self.move_camera(phi=0, theta=-90 * DEGREES, run_time=2)

        lattice_cells_2D = VGroup()

        cell_width /= 2
        for j in range(len(pdf_values_y)):
            for i in range(len(pdf_values_x)):
                # Calculate the product of the probabilities
                cell_value = pdf_values_y[-j] * pdf_values_x[i]
                # Normalize the value for color interpolation
                normalized_value = cell_value / (
                    pdf_values_y.max() * pdf_values_x.max()
                )
                cell_color = interpolate_gradient(normalized_value)
                cell = Square(
                    side_length=cell_width,
                    fill_color=cell_color,
                    fill_opacity=0.8,
                    stroke_width=0.5,
                )
                cell.move_to(
                    bar_chart_x.get_corner(DL)
                    + (0.5 * cell_width) * DOWN
                    + (0.5 * cell_width) * RIGHT
                    + (cell_width * i * RIGHT)
                    + (cell_width * j * DOWN)
                )
                lattice_cells_2D.add(cell)

        self.play(Transform(lattice_cells, lattice_cells_2D), run_time=2)

        pX = MathTex(r"p(x)", color=RED).scale(1.5).next_to(bar_chart_x, RIGHT)
        pZ = (
            MathTex(r"p(z)", color=BLUE)
            .scale(1.5)
            .next_to(bar_chart_y, LEFT, buff=0.5)
            .shift(0.8 * DOWN)
        )

        self.play(Write(pX))
        self.play(Write(pZ))

        # Display marginal distributions
        self.next_section(skip_animations=False)

        marginal_X = MathTex(r"p(x) = \int p(x, z) dz", color=RED).to_corner(
            UL, buff=0.5
        )

        self.play(Write(marginal_X))

        line = lattice_cells[15::20]
        line_ori = line.copy()

        self.wait(1)

        self.play(
            Indicate(
                bar_chart_x.bars[15],
                color=bar_chart_x.bars[15].color,
                scale_factor=1.5,
            ),
            run_time=2,
        )
        self.play(Transform(line, bar_chart_x.bars[15]), run_time=1)
        self.wait(1)
        self.play(Transform(line, line_ori), run_time=1)

        marginal_Z = MathTex(r"p(z) = \int p(x, z) dx", color=BLUE).to_corner(
            UL, buff=0.5
        )

        self.play(FadeOut(marginal_X), Write(marginal_Z))

        line = lattice_cells[300:320]
        line_ori = line.copy()
        self.play(
            Indicate(bar_chart_y.bars[4], color=bar_chart_y.bars[4].color),
            run_time=2,
        )
        self.play(Transform(line, bar_chart_y.bars[4]), run_time=1)
        self.wait(1)
        self.play(Transform(line, line_ori), run_time=1)

        self.play(FadeOut(marginal_Z))

        # Display conditional distributions
        self.next_section(skip_animations=False)

        pX_given_Z = MathTex(r"p(x|z) = \frac{p(x,z)}{p(z)}", color=RED).to_corner(
            UL, buff=0.5
        )
        self.play(Write(pX_given_Z))

        dashed_line = DashedLine(
            bar_chart_y.bars[4].get_center() + 0.2 * LEFT,
            lattice_cells[319].get_right() + 0.2 * RIGHT,
            color=WHITE,
        )

        self.play(Create(dashed_line))

        z_value = MathTex(r"z = 3").next_to(dashed_line, RIGHT, buff=0.5)

        self.play(Write(z_value))

        self.play(FadeOut(dashed_line))

        self.wait(2)
        self.play(
            LaggedStart(
                *[
                    Indicate(cell, scale_factor=2, color=RED)
                    for cell in lattice_cells[300:320]
                ],
                lag_ratio=0.3
            ),
            run_time=2,
        )

        self.play(
            Indicate(
                bar_chart_y.bars[4],
                color=bar_chart_y.bars[4].color,
                scale_factor=2.5,
            )
        )

        pZ_given_X = MathTex(r"p(z|x) = \frac{p(x,z)}{p(x)}", color=BLUE).move_to(
            pX_given_Z
        )

        self.play(FadeOut(z_value, pX_given_Z))

        self.play(Write(pZ_given_X))

        dashed_line = DashedLine(
            bar_chart_x.bars[15].get_center() + 0.2 * UP,
            lattice_cells[395].get_bottom() + 0.2 * DOWN,
            color=WHITE,
        )

        self.play(Create(dashed_line))

        x_value = MathTex(r"x = 7").next_to(dashed_line, DOWN, buff=0.1)

        self.play(Write(x_value))
        self.play(FadeOut(dashed_line))

        self.play(
            LaggedStart(
                *[
                    Indicate(cell, scale_factor=2, color=BLUE)
                    for cell in lattice_cells[15::20]
                ],
                lag_ratio=0.3
            ),
            run_time=2,
        )

        self.play(
            Indicate(
                bar_chart_x.bars[15],
                color=bar_chart_x.bars[15].color,
                scale_factor=2.5,
            )
        )

        self.play(
            FadeOut(
                pZ_given_X, x_value, bar_chart_x, bar_chart_y, lattice_cells, pX, pZ
            )
        )

        self.wait(2)


if __name__ == "__main__":
    scene = Scene2_2()
    scene.render()
