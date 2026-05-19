import numpy as np
import raylib as rl
from ai4animation import Utility


class Color:
    def GetColor(R, G, B, A=1.0):
        return (int(R * 255), int(G * 255), int(B * 255), int(A * 255))

    def GetRainbowColor(index, total):
        frequency = 5.0 / total
        return Color.GetColor(
            Utility.Normalize(
                np.sin(frequency * index + 0) * (127) + 128, 0, 255, 0, 1
            ),
            Utility.Normalize(
                np.sin(frequency * index + 2) * (127) + 128, 0, 255, 0, 1
            ),
            Utility.Normalize(
                np.sin(frequency * index + 4) * (127) + 128, 0, 255, 0, 1
            ),
            1,
        )

    def GetRainbowColors(total):
        colors = []
        for i in range(total):
            colors.append(Color.GetRainbowColor(i, total))
        return colors

    BLACK = rl.colors.BLACK
    WHITE = rl.colors.WHITE
    RED = rl.colors.RED
    GREEN = rl.colors.GREEN
    BLUE = rl.colors.BLUE
    RAYWHITE = rl.colors.RAYWHITE
    GRAY = rl.colors.GRAY
    LIGHTGRAY = rl.colors.LIGHTGRAY
    SKYBLUE = rl.colors.SKYBLUE
    ORANGE = rl.colors.ORANGE
    MAGENTA = rl.colors.MAGENTA
    CYAN = GetColor(0, 1, 1, 1)
