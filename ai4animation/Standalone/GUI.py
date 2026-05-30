# Copyright (c) Meta Platforms, Inc. and affiliates.
import builtins

import numpy as np
import raylib as rl
from ai4animation import Time, Utility
from ai4animation.AI4Animation import AI4Animation
from ai4animation.Math import Rotation, Vector3

PRECISION = 3


def ScreenWidth():
    return rl.GetScreenWidth()


def ScreenHeight():
    return rl.GetScreenHeight()


class Canvas:
    def __init__(self, label, x, y, w, h, scale_width=True, scale_height=True):
        self.Label = label
        self.Rectangle = Rectangle(x, y, w, h)
        self.Header = Rectangle(x, y, w, 0.03)
        self.Items = []
        self.ScaleWidth = scale_width
        self.ScaleHeight = scale_height
        self.HeadColor = Utility.Opacity(AI4Animation.Color.BLACK, 1.0)
        self.BodyColor = Utility.Opacity(AI4Animation.Color.BLACK, 0.3)
        self.TextColor = AI4Animation.Color.WHITE
        self.FrameColor = AI4Animation.Color.WHITE

    def AddItem(self, item):
        self.Items.append(item)

    def ToCanvas(self, rectangle):
        rectangle = rectangle.Copy()

        if self.ScaleWidth:
            rectangle.x = self.Rectangle.x + self.Rectangle.width * rectangle.x
            rectangle.width *= self.Rectangle.width
        else:
            rectangle.x += self.Rectangle.x

        if self.ScaleHeight:
            rectangle.y = self.Rectangle.y + self.Rectangle.height * rectangle.y
            rectangle.height *= self.Rectangle.height
        else:
            rectangle.y += self.Rectangle.y

        return rectangle

    def GUI(self):
        rl.DrawRectangleRec(self.Rectangle.Screen().Tuple(), self.BodyColor)
        rl.DrawRectangleRec(self.Header.Screen().Tuple(), self.HeadColor)
        rl.DrawRectangleLinesEx(self.Rectangle.Screen().Tuple(), 2.0, self.FrameColor)

        size = self.Header.height * 0.75
        offset_x = (
            self.Header.width / 2
            - rl.MeasureText(Utility.ToBytes(self.Label), int(ScreenHeight() * size))
            / ScreenWidth()
            / 2
        )
        offset_y = self.Header.height / 2 - 0.4 * size
        AI4Animation.Draw.Text(
            self.Label,
            self.Rectangle.x + offset_x,
            self.Rectangle.y + offset_y,
            size,
            color=self.TextColor,
        )

        for item in self.Items:
            item.GUI()


class Slider:
    def __init__(
        self, x, y, w, h, value, min, max, integers=False, canvas=None, label=None
    ):
        self.Rectangle = Rectangle(x, y, w, h)
        self.Value = rl.ffi.new("float *", value)
        self.Min = min
        self.Max = max
        self.Integers = integers
        self.Canvas = canvas
        self.PrevValue = self.GetValue()
        self.Modified = False
        self.Label = label

    def GUI(self, value=None):
        rectangle = (
            self.Canvas.ToCanvas(self.Rectangle) if self.Canvas else self.Rectangle
        )
        rl.GuiSliderBar(
            rectangle.Screen().Tuple(), b"", b"", self.Value, self.Min, self.Max
        )
        self.Modified = round(self.GetValue(), PRECISION) != round(
            self.PrevValue, PRECISION
        )
        self.PrevValue = self.GetValue()
        if value is not None and not self.Modified:
            self.SetValue(value)

        if self.Label is not None:
            AI4Animation.Draw.Text(
                self.Label + ": " + str(round(self.GetValue(), PRECISION)),
                rectangle.x + rectangle.width / 2.0,
                rectangle.y + rectangle.height / 4.0,
                size=rectangle.height / 2.0,
                color=AI4Animation.Color.BLACK,
                pivot=0.5,
            )

        return self.GetValue()

    def SetValue(self, value):
        value = int(round(value)) if self.Integers else value
        self.Value[0] = value
        self.PrevValue = value

    def GetValue(self):
        return int(round(self.Value[0])) if self.Integers else self.Value[0]


class Dropdown:
    def __init__(
        self, label, x, y, w, h, options, canvas=None
    ):  # Option is tuple of (name, lambda)
        self.Rectangle = Rectangle(x, y, w, h)
        self.Canvas = canvas
        self.Button = Button(label, x, y, w, h, False, True, canvas)
        self.Options = options
        self.Items = None

    def Select(self, index):
        self.Options[index][1](index)
        self.Button.Active = False
        self.Items = None

    def GUI(self, colors=None):
        self.Button.GUI()
        if self.Button.Active:
            if self.Items is None:
                self.Items = []
                for i in range(len(self.Options)):
                    self.Items.append(
                        Button(
                            self.Options[i][0],
                            self.Rectangle.x,
                            self.Rectangle.y + (i + 1) * self.Rectangle.height,
                            self.Rectangle.width,
                            self.Rectangle.height,
                            False,
                            True,
                            canvas=self.Canvas,
                            color_default=(
                                AI4Animation.Color.ORANGE
                                if colors is not None and colors[i]
                                else None
                            ),
                            color_hovered=(
                                AI4Animation.Color.ORANGE
                                if colors is not None and colors[i]
                                else None
                            ),
                        )
                    )
            for i in range(len(self.Items)):
                self.Items[i].GUI()
                if self.Items[i].Active:
                    self.Select(i)
                    break


class TextField:
    def __init__(self, x, y, w, h, canvas=None, default=None):
        self.Text = ""
        self.Rectangle = Rectangle(x, y, w, h)
        self.Changed = False
        self.Selected = False
        self.Canvas = canvas
        self.Default = "" if default is None else default

    def IsHovered(self):
        rectangle = (
            self.Canvas.ToCanvas(self.Rectangle) if self.Canvas else self.Rectangle
        )
        return rl.CheckCollisionPointRec(
            rl.GetMousePosition(), rectangle.Screen().Tuple()
        )

    def BecomesSelected(self):
        return (
            not self.Selected
            and self.IsHovered()
            and rl.IsMouseButtonPressed(rl.MOUSE_BUTTON_LEFT)
        )

    def BecomesReleased(self):
        return (
            self.Selected
            and not self.IsHovered()
            and rl.IsMouseButtonReleased(rl.MOUSE_BUTTON_LEFT)
        )

    def GUI(self):
        rectangle = (
            self.Canvas.ToCanvas(self.Rectangle) if self.Canvas else self.Rectangle
        )
        screenRectangle = rectangle.Screen()
        rl.DrawRectangleRec(screenRectangle.Tuple(), AI4Animation.Color.WHITE)
        rl.DrawRectangleLinesEx(
            screenRectangle.Tuple(),
            2.0,
            (
                AI4Animation.Color.BLACK
                if self.Selected
                else (
                    AI4Animation.Color.LIGHTGRAY
                    if self.IsHovered()
                    else AI4Animation.Color.WHITE
                )
            ),
        )

        if self.BecomesSelected():
            self.Selected = True

        if self.BecomesReleased():
            self.Selected = False

        self.Changed = False
        if self.Selected:
            key = AI4Animation.Standalone.IO.GetCurrentKey()
            if key is not None:
                self.Text += key
                self.Changed = True
            if AI4Animation.Standalone.IO.GetKey(rl.KEY_BACKSPACE):
                self.Text = self.Text[:-1]
                self.Changed = True

        if self.Text == "" and not self.Selected:
            AI4Animation.Draw.Text(
                self.Default,
                rectangle.x + 0.01 * rectangle.width,
                rectangle.y + 0.2 * rectangle.height,
                size=0.7 * rectangle.height,
                color=AI4Animation.Color.LIGHTGRAY,
            )
        else:
            text = self.Text
            if self.Selected:
                if Time.TotalTime % 1.0 < 0.5:
                    text += "_"
            AI4Animation.Draw.Text(
                text,
                rectangle.x + 0.01 * rectangle.width,
                rectangle.y + 0.2 * rectangle.height,
                size=0.7 * rectangle.height,
                color=AI4Animation.Color.BLACK,
            )


class Button:
    def __init__(
        self,
        label,
        x,
        y,
        w,
        h,
        state=True,
        toggle=True,
        canvas=None,
        color_default=None,
        color_hovered=None,
        color_active=None,
        border_default=None,
        border_hovered=None,
        border_active=None,
    ):
        self.Label = label
        self.Rectangle = Rectangle(x, y, w, h)
        self.Active = state
        self.Canvas = canvas
        self.Toggle = toggle

        self.ColorDefault = (
            AI4Animation.Color.LIGHTGRAY if not color_default else color_default
        )
        self.ColorHovered = (
            AI4Animation.Color.LIGHTGRAY if not color_hovered else color_hovered
        )
        self.ColorActive = (
            AI4Animation.Color.SKYBLUE if not color_active else color_active
        )
        self.BorderDefault = (
            AI4Animation.Color.LIGHTGRAY if not border_default else border_default
        )
        self.BorderHovered = (
            AI4Animation.Color.BLACK if not border_hovered else border_hovered
        )
        self.BorderActive = (
            AI4Animation.Color.BLACK if not border_active else border_active
        )

    def IsHovered(self):
        rectangle = (
            self.Canvas.ToCanvas(self.Rectangle) if self.Canvas else self.Rectangle
        )
        return rl.CheckCollisionPointRec(
            rl.GetMousePosition(), rectangle.Screen().Tuple()
        )

    def IsPressed(self):
        return self.IsHovered() and rl.IsMouseButtonReleased(rl.MOUSE_BUTTON_LEFT)

    def GUI(self):
        if self.Toggle:
            if self.IsPressed() and rl.IsMouseButtonReleased(rl.MOUSE_BUTTON_LEFT):
                self.Active = not self.Active

        rectangle = (
            self.Canvas.ToCanvas(self.Rectangle) if self.Canvas else self.Rectangle
        )
        screenRectangle = rectangle.Screen()
        rl.DrawRectangleRec(
            screenRectangle.Tuple(),
            (
                self.ColorActive
                if self.Active
                else self.ColorHovered
                if self.IsHovered()
                else self.ColorDefault
            ),
        )
        rl.DrawRectangleLinesEx(
            screenRectangle.Tuple(),
            2.0,
            (
                self.BorderActive
                if self.Active
                else self.BorderHovered
                if self.IsHovered()
                else self.BorderDefault
            ),
        )

        size = rectangle.height * 0.75
        offset_x = (
            rectangle.width / 2
            - rl.MeasureText(Utility.ToBytes(self.Label), int(ScreenHeight() * size))
            / ScreenWidth()
            / 2
        )
        offset_y = rectangle.height / 2 - size / 2
        AI4Animation.Draw.Text(
            self.Label, rectangle.x + offset_x, rectangle.y + offset_y, size
        )


class Handle:
    def __init__(
        self, entity, defaultColor=None, hoveredColor=None, selectedColor=None
    ):
        self.Entity = entity
        self.Rectangle = Rectangle(0, 0, 0, 0)
        self.Selected = False
        self.SelectedRotate = False
        self.DefaultColor = (
            AI4Animation.Color.GRAY if not defaultColor else defaultColor
        )
        self.HoveredColor = (
            AI4Animation.Color.BLACK if not hoveredColor else hoveredColor
        )
        self.SelectedColor = (
            AI4Animation.Color.RED if not selectedColor else selectedColor
        )

    def GUI(self):
        width = 0.0125
        height = width * ScreenWidth() / ScreenHeight()
        pos = rl.GetWorldToScreen(
            Vector3.ToRayLib(self.Entity.GetPosition()),
            AI4Animation.Standalone.Camera.Camera,
        )
        self.Rectangle = Rectangle(
            pos.x / ScreenWidth() - width / 2,
            pos.y / ScreenHeight() - height / 2,
            width,
            height,
        )

        if self.IsHovered():
            rl.DrawCircle(
                int(
                    self.Rectangle.Screen().Tuple()[0]
                    + self.Rectangle.Screen().Tuple()[2] / 2
                ),
                int(
                    self.Rectangle.Screen().Tuple()[1]
                    + self.Rectangle.Screen().Tuple()[3] / 2
                ),
                int(self.Rectangle.Screen().Tuple()[2] / 2),
                Utility.Opacity(
                    self.SelectedColor if self.Selected else self.HoveredColor, 0.5
                ),
            )
        else:
            rl.DrawCircle(
                int(
                    self.Rectangle.Screen().Tuple()[0]
                    + self.Rectangle.Screen().Tuple()[2] / 2
                ),
                int(
                    self.Rectangle.Screen().Tuple()[1]
                    + self.Rectangle.Screen().Tuple()[3] / 2
                ),
                int(self.Rectangle.Screen().Tuple()[2] / 2),
                Utility.Opacity(self.DefaultColor, 0.5),
            )

            # rl.DrawRectangleRec(self.Rectangle.Screen().Tuple(), Opacity(RED if self.IsSelected() else BLACK, 0.5))

        if self.BecomesSelected():
            self.Selected = True

        if self.BecomesSelectedRotate():
            self.SelectedRotate = True

        if self.BecomesReleased():
            self.Selected = False

        if self.BecomesReleasedRotate():
            self.SelectedRotate = False

        if self.Selected:
            position = AI4Animation.Standalone.IO.GetMousePositionInSpace(
                AI4Animation.Standalone.Camera, self.Entity
            )
            if Vector3.Length(position) != 0.0:
                self.Entity.SetPosition(position)

        if self.SelectedRotate:
            rate = 30.0
            dt = Time.DeltaTime
            dx, dy = AI4Animation.Standalone.IO.GetMouseDeltaOnScreen()
            self.Entity.SetRotation(
                Rotation.Multiply(
                    self.Entity.GetRotation(),
                    Rotation.Euler(dy * rate * dt, dx * rate * dt, 0.0),
                )
            )

    def IsHovered(self):
        return rl.CheckCollisionPointRec(
            rl.GetMousePosition(), self.Rectangle.Screen().Tuple()
        )

    def BecomesSelected(self):
        return (
            not self.Selected
            and self.IsHovered()
            and rl.IsMouseButtonPressed(rl.MOUSE_BUTTON_LEFT)
        )

    def BecomesReleased(self):
        return self.Selected and rl.IsMouseButtonReleased(rl.MOUSE_BUTTON_LEFT)

    def BecomesSelectedRotate(self):
        return (
            not self.SelectedRotate
            and self.IsHovered()
            and rl.IsMouseButtonPressed(rl.MOUSE_BUTTON_RIGHT)
        )

    def BecomesReleasedRotate(self):
        return self.SelectedRotate and rl.IsMouseButtonReleased(rl.MOUSE_BUTTON_RIGHT)


class Rectangle:
    def __init__(self, x, y, width, height):
        self.x = x
        self.y = y
        self.width = width
        self.height = height

    def Screen(self):
        x = ScreenWidth() * self.x
        y = ScreenHeight() * self.y
        w = ScreenWidth() * self.width
        h = ScreenHeight() * self.height
        return Rectangle(x, y, w, h)

    def Tuple(self):
        def to_scalar(v):
            if hasattr(v, "item"):
                return v.item()
            return float(v)
        return (to_scalar(self.x), to_scalar(self.y), to_scalar(self.width), to_scalar(self.height))

    def Copy(self):
        return Rectangle(self.x, self.y, self.width, self.height)


def _clamp_value(value, min_val, max_val):
    if min_val is not None and value < min_val:
        return min_val
    if max_val is not None and value > max_val:
        return max_val
    return value


def _draw_curve_legend(
    curveLabels, colors, plot_left, plot_right, label_height, legend_font_size, y, h
):
    legend_row_y = y + 0.015 + label_height
    marker_size = 0.012
    marker_gap = 0.006
    legend_count = len(curveLabels)
    slot_width = (plot_right - plot_left) / legend_count
    for idx, curveLabel in enumerate(curveLabels):
        text_width = (
            rl.MeasureText(Utility.ToBytes(curveLabel), legend_font_size)
            / ScreenWidth()
        )
        item_width = marker_size + marker_gap + text_width
        slot_left = plot_left + idx * slot_width
        item_x = slot_left + (slot_width - item_width) / 2.0
        marker = Rectangle(
            item_x, legend_row_y + 0.002, marker_size, marker_size
        ).Screen()
        rl.DrawRectangleRec(marker.Tuple(), colors[idx % len(colors)])
        AI4Animation.Draw.Text(
            curveLabel,
            item_x + marker_size + marker_gap,
            legend_row_y,
            size=h / 9.0,
            color=AI4Animation.Color.BLACK,
        )


def CurvePlot(
    x,
    y,
    w,
    h,
    values,
    label=None,
    min=None,
    max=None,
    colors=None,
    curveLabels=None,
    backgroundColor=None,
    frameColor=None,
):
    if len(values.shape) > 3:
        print(
            "Drawing curve plot for tensors with more than 2 dimensions is not supported."
        )
        return

    if len(values.shape) == 1:
        values = values.reshape(1, -1)

    rectangle = Rectangle(x, y, w, h)
    screenRectangle = rectangle.Screen()
    rows = values.shape[0]
    cols = values.shape[1]

    if cols < 2:
        return

    backgroundColor = (
        Utility.Opacity(AI4Animation.Color.WHITE, 0.5)
        if backgroundColor is None
        else backgroundColor
    )
    frameColor = AI4Animation.Color.BLACK if frameColor is None else frameColor
    colors = (
        [
            AI4Animation.Color.RED,
            AI4Animation.Color.BLUE,
            AI4Animation.Color.GREEN,
            AI4Animation.Color.ORANGE,
            AI4Animation.Color.PURPLE,
            AI4Animation.Color.BLACK,
        ]
        if colors is None
        else colors
    )

    values_min = float(np.min(values)) if min is None else min
    values_max = float(np.max(values)) if max is None else max
    if values_min == values_max:
        values_max = values_min + 1.0

    rl.DrawRectangleRec(screenRectangle.Tuple(), backgroundColor)

    label_font_size = builtins.max(10, int(ScreenHeight() * (h / 6.0)))
    legend_font_size = builtins.max(10, int(ScreenHeight() * (h / 9.0)))
    label_height = 0.0 if label is None else label_font_size / ScreenHeight() + 0.01

    legend_count = 0
    if curveLabels is not None:
        legend_count = len(curveLabels)

    legend_height = (
        0.0 if legend_count == 0 else legend_font_size / ScreenHeight() + 0.016
    )
    top_padding = 0.025 * h + label_height + legend_height
    left_padding = 0.03 * w
    right_padding = 0.02 * w
    bottom_padding = 0.05 * h

    plot_left = x + left_padding
    plot_right = x + w - right_padding
    plot_top = y + top_padding
    plot_bottom = y + h - bottom_padding

    grid_color = Utility.Opacity(frameColor, 0.25)
    for step in range(1, 4):
        grid_y = Utility.Normalize(step, 0, 4, plot_bottom, plot_top)
        rl.DrawLine(
            int(ScreenWidth() * plot_left),
            int(ScreenHeight() * grid_y),
            int(ScreenWidth() * plot_right),
            int(ScreenHeight() * grid_y),
            grid_color,
        )

    for row in range(rows):
        color = colors[row % len(colors)]
        for col in range(1, cols):
            prev_value = _clamp_value(values[row, col - 1], min, max)
            next_value = _clamp_value(values[row, col], min, max)

            x1 = ScreenWidth() * Utility.Normalize(
                col - 1, 0, cols - 1, plot_left, plot_right
            )
            x2 = ScreenWidth() * Utility.Normalize(
                col, 0, cols - 1, plot_left, plot_right
            )
            y1 = ScreenHeight() * Utility.Normalize(
                prev_value, values_min, values_max, plot_bottom, plot_top
            )
            y2 = ScreenHeight() * Utility.Normalize(
                next_value, values_min, values_max, plot_bottom, plot_top
            )
            rl.DrawLine(int(x1), int(y1), int(x2), int(y2), color)

        final_value = _clamp_value(values[row, -1], min, max)
        final_x = ScreenWidth() * plot_right
        final_y = ScreenHeight() * Utility.Normalize(
            final_value, values_min, values_max, plot_bottom, plot_top
        )
        rl.DrawCircle(int(final_x), int(final_y), 3.0, color)

    rl.DrawRectangleLinesEx(screenRectangle.Tuple(), 2.0, frameColor)

    if label is not None:
        AI4Animation.Draw.Text(
            label,
            x + w / 2.0,
            y + 0.015,
            size=h / 6.0,
            color=AI4Animation.Color.BLACK,
            pivot=0.5,
        )

    if legend_count > 0:
        _draw_curve_legend(
            curveLabels,
            colors,
            plot_left,
            plot_right,
            label_height,
            legend_font_size,
            y,
            h,
        )


def BarPlot(
    x,
    y,
    w,
    h,
    values,
    label=None,
    min=None,
    max=None,
    colors=None,
    backgroundColor=None,
    frameColor=None,
):
    if len(values.shape) > 3:
        print(
            "Drawing bar plot for tensors with more than 2 dimensions is not supported."
        )
        return

    if len(values.shape) == 1:
        values = values.reshape(1, -1)

    rectangle = Rectangle(x, y, w, h)
    screenRectangle = rectangle.Screen()

    backgroundColor = (
        Utility.Opacity(AI4Animation.Color.WHITE, 0.5)
        if backgroundColor is None
        else backgroundColor
    )
    frameColor = AI4Animation.Color.BLACK if frameColor is None else frameColor
    if colors is None:
        colors = [AI4Animation.Color.BLACK]

    rl.DrawRectangleRec(screenRectangle.Tuple(), backgroundColor)

    rows = values.shape[0]
    cols = values.shape[1]

    bar_width = w / cols
    bar_height = h / rows
    row_gap = bar_height * 0.15
    bar_inner_height = bar_height - row_gap
    for row in range(rows):
        color = colors[row % len(colors)]
        for col in range(cols):
            value = values[row, col]
            if min is not None and max is not None:
                value = _clamp_value(value, min, max)
                value = Utility.Normalize(value, min, max, 0.0, 1.0)
            ratio = bar_inner_height * value
            bar_x = ScreenWidth() * (
                (x + w - bar_width)
                if cols == 1
                else Utility.Normalize(col, 0, cols - 1, x, x + w - bar_width)
            )
            bar_y = ScreenHeight() * (
                (y + h - ratio)
                if rows == 1
                else Utility.Normalize(
                    row, 0, rows - 1, y + bar_height - ratio, y + h - ratio
                )
            )
            bar_w = ScreenWidth() * bar_width
            bar_h = ScreenHeight() * ratio
            rl.DrawRectangleRec((bar_x, bar_y, bar_w, bar_h), color)

    rl.DrawRectangleLinesEx(screenRectangle.Tuple(), 2.0, frameColor)

    if label is not None:
        AI4Animation.Draw.Text(
            label,
            x + w / 2.0,
            y + h / 4.0,
            size=h / 2.0,
            color=AI4Animation.Color.BLACK,
            pivot=0.5,
        )


def HorizontalPivot(
    x,
    y,
    w,
    h,
    value,
    label=None,
    limits=None,
    thickness=0.025,
    backgroundColor=None,
    pivotColor=None,
):
    if limits is not None:
        value = Utility.Normalize(value, limits[0], limits[1], 0.0, 1.0)
    frameRectangle = Rectangle(x, y, w, h).Screen()
    pivotRectangle = Rectangle(
        x + value * (w - thickness / 2.0), y, w * thickness, h
    ).Screen()
    backgroundColor = (
        Utility.Opacity(AI4Animation.Color.WHITE, 0.5)
        if backgroundColor is None
        else backgroundColor
    )
    pivotColor = AI4Animation.Color.WHITE if pivotColor is None else pivotColor
    rl.DrawRectangleRec(frameRectangle.Tuple(), backgroundColor)
    rl.DrawRectangleRec(pivotRectangle.Tuple(), pivotColor)
    rl.DrawRectangleLinesEx(frameRectangle.Tuple(), 2.0, AI4Animation.Color.BLACK)
    if label is not None:
        AI4Animation.Draw.Text(
            label,
            x + w / 2.0,
            y + h / 4.0,
            size=h / 2.0,
            color=AI4Animation.Color.BLACK,
            pivot=0.5,
        )


def HorizontalBar(
    x, y, w, h, value, label=None, limits=None, backgroundColor=None, pivotColor=None
):
    if limits is not None:
        value = Utility.Normalize(value, limits[0], limits[1], 0.0, 1.0)
    frameRectangle = Rectangle(x, y, w, h).Screen()
    pivotRectangle = Rectangle(x, y, value * w, h).Screen()
    backgroundColor = (
        Utility.Opacity(AI4Animation.Color.BLACK, 0.5)
        if backgroundColor is None
        else backgroundColor
    )
    pivotColor = AI4Animation.Color.WHITE if pivotColor is None else pivotColor
    rl.DrawRectangleRec(frameRectangle.Tuple(), backgroundColor)
    rl.DrawRectangleRec(pivotRectangle.Tuple(), pivotColor)
    rl.DrawRectangleLinesEx(frameRectangle.Tuple(), 2.0, AI4Animation.Color.BLACK)
    if label is not None:
        AI4Animation.Draw.Text(
            label,
            x + w / 2.0,
            y + h / 4.0,
            size=h / 2.0,
            color=AI4Animation.Color.BLACK,
            pivot=0.5,
        )
