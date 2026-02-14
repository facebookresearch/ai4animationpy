# Copyright (c) Meta Platforms, Inc. and affiliates.
import os
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

    def __init__(self, program, mode=Mode.STANDALONE, profiler=None):
        AI4Animation.Program = program
        AI4Animation.RunMode = mode
        AI4Animation.Profiler = profiler

        AI4Animation.Standalone = None
        AI4Animation.Scene = Scene.Scene()
        AI4Animation.Draw = None
        AI4Animation.GUI = None
        AI4Animation.Color = None

        # Load Standalone
        if mode == self.Mode.STANDALONE:
            Utility.LoadModule(
                os.path.dirname(__file__) + "/Standalone/Standalone.py"
            ).Standalone()

        # Initialize Scene
        if mode == self.Mode.STANDALONE:
            if hasattr(AI4Animation.Scene, "Standalone"):
                AI4Animation.Scene.Standalone()

        # Initialize Program
        if hasattr(AI4Animation.Program, "Start"):
            AI4Animation.Program.Start()
        if mode == self.Mode.STANDALONE:
            if hasattr(AI4Animation.Program, "Standalone"):
                AI4Animation.Program.Standalone()

        # Run Update Loop
        if mode == self.Mode.STANDALONE:
            AI4Animation.Standalone.Run()
        if mode == self.Mode.HEADLESS:
            then = time.time()
            while True:
                now = time.time()
                dt = now - then
                then = now
                if dt > 0.0:
                    AI4Animation.Update(dt)
        if mode == self.Mode.MANUAL:
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
