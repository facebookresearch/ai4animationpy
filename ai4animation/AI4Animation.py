# Copyright (c) Meta Platforms, Inc. and affiliates.
"""Core application class managing the update loop, rendering, and module loading."""

import os
import sys
import time
from enum import Enum

from ai4animation import Scene, Time, Utility


class AI4Animation:
    global Program
    global RunMode
    global Profiler

    global Standalone
    global Scene
    global Draw
    global GUI
    global Color

    class Mode(Enum):
        STANDALONE = 1
        HEADLESS = 2
        MANUAL = 3

    def IsStandalone():
        return AI4Animation.RunMode == AI4Animation.Mode.STANDALONE

    def IsHeadless():
        return AI4Animation.RunMode == AI4Animation.Mode.HEADLESS

    def IsManual():
        return AI4Animation.RunMode == AI4Animation.Mode.MANUAL

    def DetermineRunMode():
        if AI4Animation.HasDisplay():
            return AI4Animation.Mode.STANDALONE
        else:
            return AI4Animation.Mode.HEADLESS

    def __init__(self, program, mode=None, profiler=None):
        AI4Animation.Program = program
        AI4Animation.RunMode = (
            mode if mode is not None else AI4Animation.DetermineRunMode()
        )
        AI4Animation.Profiler = profiler

        AI4Animation.Standalone = None
        AI4Animation.Scene = Scene.Scene()
        AI4Animation.Draw = None
        AI4Animation.GUI = None
        AI4Animation.Color = None

        # Load Standalone
        if AI4Animation.RunMode == self.Mode.STANDALONE:
            Utility.LoadModule(
                os.path.dirname(__file__) + "/Standalone/Standalone.py"
            ).Standalone()

        # Initialize Scene
        if AI4Animation.RunMode == self.Mode.STANDALONE:
            if hasattr(AI4Animation.Scene, "Standalone"):
                AI4Animation.Scene.Standalone()

        # Initialize Program
        if hasattr(AI4Animation.Program, "Start"):
            AI4Animation.Program.Start()
        if AI4Animation.RunMode == self.Mode.STANDALONE:
            if hasattr(AI4Animation.Program, "Standalone"):
                AI4Animation.Program.Standalone()

        # Run Update Loop
        if AI4Animation.RunMode == self.Mode.STANDALONE:
            AI4Animation.Standalone.Run()
        if AI4Animation.RunMode == self.Mode.HEADLESS:
            then = time.time()
            while True:
                now = time.time()
                dt = now - then
                then = now
                if dt > 0.0:
                    AI4Animation.Update(dt)
        if AI4Animation.RunMode == self.Mode.MANUAL:
            pass

    @staticmethod
    def Update(deltaTime):
        Time.DeltaTime = deltaTime * Time.Timescale
        Time.TotalTime += Time.DeltaTime
        if AI4Animation.Standalone is not None:
            AI4Animation.Standalone.Update()
        else:
            AI4Animation.__UPDATE__()

    @staticmethod
    def __UPDATE__():
        if hasattr(AI4Animation.Program, "Update"):
            AI4Animation.Program.Update()
        AI4Animation.Scene.Update()

        if AI4Animation.Profiler:
            AI4Animation.Profiler.Check()

    @staticmethod
    def __DRAW__():
        if AI4Animation.Standalone is not None:
            if hasattr(AI4Animation.Program, "Draw"):
                AI4Animation.Program.Draw()
            AI4Animation.Scene.Draw()

    @staticmethod
    def __GUI__():
        if AI4Animation.Standalone is not None:
            if hasattr(AI4Animation.Program, "GUI"):
                AI4Animation.Program.GUI()
            AI4Animation.Scene.GUI()

    def HasDisplay() -> bool:
        if sys.platform.startswith("win"):
            # Windows: use ctypes to query monitor count
            import ctypes

            user32 = ctypes.windll.user32
            # SM_CMONITORS = 80
            return user32.GetSystemMetrics(80) > 0

        if sys.platform == "darwin":
            # macOS: a desktop session always has a display unless SSH-only.
            # Use Quartz if available; otherwise assume yes.
            try:
                import ctypes

                from Quartz import CGGetActiveDisplayList  # type: ignore

                max_displays = 16
                active = (ctypes.c_uint32 * max_displays)()
                count = ctypes.c_uint32(0)
                CGGetActiveDisplayList(max_displays, active, ctypes.byref(count))
                return count.value > 0
            except ImportError:
                return True

        # Linux / *nix: rely on environment variables set by the display server
        return bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))
