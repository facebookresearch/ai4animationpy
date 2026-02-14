# Copyright (c) Meta Platforms, Inc. and affiliates.
import os
from datetime import datetime

import pyscreenrec
from ai4animation import Time
from ai4animation.AI4Animation import AI4Animation
from ai4animation.Components.Component import Component


class VideoRecorder(Component):
    def Start(self, params):
        self.Instance = None
        self.Duration = 0.0
        self.IsRecording = False

    def StartRecording(self, filename=None, directory=None, fps=30):
        if self.IsRecording:
            print("VideoRecorder: Already recording")

        try:
            directory = directory or "recordings/"
            if filename is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"recording_{timestamp}.mp4"
            else:
                filename = f"{filename}.mp4"
            # Ensure output directory exists
            os.makedirs(directory, exist_ok=True)
            full_path = os.path.join(directory, filename)

            # Initialize and start recording
            self.Instance = pyscreenrec.ScreenRecorder()
            XY = AI4Animation.Standalone.WindowPosition()
            W = AI4Animation.Standalone.ScreenWidth()
            H = AI4Animation.Standalone.ScreenHeight()

            self.Instance.start_recording(
                full_path,
                fps,
                {"mon": 1, "left": XY[0], "top": XY[1], "width": W, "height": H},
            )

            self.Duration = 0.0
            self.IsRecording = True
            print(f"Started Video Recording: {full_path}")

        except Exception as e:
            print(f"VideoRecorder Error: Failed to start recording - {str(e)}")
            self.Instance = None
            self.IsRecording = False

    def StopRecording(self):
        if self.IsRecording and self.Instance:
            self.Instance.stop_recording()
            self.Instance = None
            self.IsRecording = False
            print(f"Stopped Video Recording (Duration: {self.Duration:.2f}s)")
        else:
            print("VideoRecorder: No recording in progress")

    def Update(self):
        if self.Button_Record.IsPressed():
            if self.IsRecording:
                self.StopRecording()
            else:
                self.StartRecording()

        # Update duration
        if self.IsRecording:
            self.Duration += Time.DeltaTime

    def Standalone(self):
        self.Button_Record = AI4Animation.GUI.Button(
            "Record", 0.0125, 0.89, 0.1, 0.05, False, True
        )

    def GUI(self):
        if self.IsRecording:
            AI4Animation.Draw.Text(
                f"Recording: {self.Duration:.1f}s",
                0.0125,
                0.86,
                0.02,
                AI4Animation.Color.RED,
            )
        self.Button_Record.GUI()
