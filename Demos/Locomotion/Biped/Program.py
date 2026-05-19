# Copyright (c) Meta Platforms, Inc. and affiliates.
import os
import sys
from pathlib import Path

import numpy as np
import raylib as rl
import torch

SCRIPT_DIR = Path(__file__).parent
ASSETS_PATH = str(SCRIPT_DIR.parent.parent / "_ASSETS_/Geno")

sys.path.append(ASSETS_PATH)
import Definitions
from ai4animation import (
    Actor,
    AI4Animation,
    FABRIK,
    FeedTensor,
    GuidanceModule,
    MotionModule,
    ReadTensor,
    RootModule,
    Rotation,
    Tensor,
    Time,
    TimeSeries,
    Transform,
    Utility,
    Vector3,
)
from LegIK import LegIK
from Sequence import Sequence

MIN_TIMESCALE = 1.0
MAX_TIMESCALE = 1.5
SYNCHRONIZATION_SENSITIVITY = 5
TIMESCALE_SENSITIVITY = 5
SEQUENCE_WINDOW = 0.5
SEQUENCE_LENGTH = 16
SEQUENCE_FPS = 30
PREDICTION_FPS = 10
CONTACT_POWER = 3.0
CONTACT_THRESHOLD = 2.0 / 3.0

# The locomotion system was trained on the Style100 dataset.


class Program:
    def Start(self):
        self.Actor = AI4Animation.Scene.AddEntity("Actor").AddComponent(
            Actor,
            os.path.join(ASSETS_PATH, "Model.glb"),
            Definitions.FULL_BODY_NAMES,
            True,
        )
        AI4Animation.Standalone.Camera.SetTarget(self.Actor.Entity)

        local_path = os.path.join(SCRIPT_DIR, "Network.pt")
        self.Model = torch.load(local_path, weights_only=False)

        self.Model.eval()

        self.SolverIterations = 1
        self.SolverAccuracy = 1e-3

        self.NetworkIterations = 3

        self.Synchronization = 0.0
        self.Timescale = 1.0

        self.TrajectoryCorrection = 0.25
        self.GuidanceCorrection = 0.0

        self.ControlSeries = TimeSeries(0.0, SEQUENCE_WINDOW, SEQUENCE_LENGTH)
        self.SimulationObject = RootModule.Series(self.ControlSeries)

        self.RootControl = RootModule.Series(self.ControlSeries)
        self.GuidanceControl = GuidanceModule.Guidance(
            "Guidance", self.Actor.GetBoneNames(), self.Actor.GetPositions().copy()
        )
        self.GuidanceTemplates = {}
        directory = "Guidances"
        for path in sorted(os.listdir(directory)):
            with np.load(directory + "/" + path, allow_pickle=True) as data:
                id = Path(path).stem
                names = data["Names"]
                positions = data["Positions"]
                self.GuidanceTemplates[id] = GuidanceModule.Guidance(
                    id, names, positions
                )
                print("Added Guidance:", id)

        self.GuidanceNames = sorted(self.GuidanceTemplates.keys())

        self.GuidanceStyleIndex = 0
        self.SelectedGuidance = self.GuidanceNames[self.GuidanceStyleIndex]
        self.GuidanceControl.Positions = self.GuidanceTemplates[
            self.SelectedGuidance
        ].Positions.copy()

        self.Previous = None
        self.Sequence = None

        self.ContactBones = [
            Definitions.LeftAnkleName,
            Definitions.LeftBallName,
            Definitions.RightAnkleName,
            Definitions.RightBallName,
        ]
        self.ContactIndices = self.Actor.GetBoneIndices(self.ContactBones)

        self.LeftLegIK = LegIK(
            FABRIK(
                self.Actor.GetBone(Definitions.LeftHipName),
                self.Actor.GetBone(Definitions.LeftAnkleName),
            ),
            FABRIK(
                self.Actor.GetBone(Definitions.LeftAnkleName),
                self.Actor.GetBone(Definitions.LeftBallName),
            ),
        )

        self.RightLegIK: LegIK = LegIK(
            FABRIK(
                self.Actor.GetBone(Definitions.RightHipName),
                self.Actor.GetBone(Definitions.RightAnkleName),
            ),
            FABRIK(
                self.Actor.GetBone(Definitions.RightAnkleName),
                self.Actor.GetBone(Definitions.RightBallName),
            ),
        )

        self.Timestamp = Time.TotalTime

        AI4Animation.Standalone.IO.LogErrorIfGamepadNotAvailable()

    def _set_guidance(self, index):
        self.GuidanceStyleIndex = index % len(self.GuidanceNames)
        self.SelectedGuidance = self.GuidanceNames[self.GuidanceStyleIndex]
        self.GuidanceControl.Positions = self.GuidanceTemplates[
            self.SelectedGuidance
        ].Positions.copy()
        if hasattr(self, "GuidanceDropdown"):
            self.GuidanceDropdown.Button.Label = f"Style: {self.SelectedGuidance}"

    def Update(self):
        # Update control every frame
        self.Control()

        # Predict future sequence every few frames
        if (
            self.Timestamp == 0.0
            or Time.TotalTime - self.Timestamp > 1.0 / PREDICTION_FPS
        ):
            self.Timestamp = Time.TotalTime
            self.Predict()

        # Animate motion every frame
        self.Animate()

    def Control(self):
        # Note: raylib is used for control inputs so headless/manual mode not yet supported
        speed_sprint = 2.0
        speed_normal = 1.0
        if AI4Animation.Standalone.IO.GamepadAvailable():
            left_stick = AI4Animation.Standalone.IO.GetLeftStick()
            right_stick = AI4Animation.Standalone.IO.GetRightStick()
            speed = (
                speed_sprint
                if AI4Animation.Standalone.IO.IsLeftStickPressed()
                else speed_normal
            )

            # Handle guidance selection with L1 and R1 buttons
            if AI4Animation.Standalone.IO.IsL1Pressed():
                self._set_guidance(self.GuidanceStyleIndex - 1)
            if AI4Animation.Standalone.IO.IsR1Pressed():
                self._set_guidance(self.GuidanceStyleIndex + 1)

        # Keyboard control when no gamepad is available
        else:
            # WASD for left stick (velocity)
            # Left Shift for speed
            # Right Click and move mouse for right stick (direction)
            # Detail: We use momentum on the mouse direction start point to smooth out control over using GetMouseDeltaOnScreen() which is very noisy
            left_stick_vec3 = AI4Animation.Standalone.IO.GetWASDQE()
            left_stick = [left_stick_vec3[0], left_stick_vec3[2]]
            speed = speed_sprint if rl.IsKeyDown(rl.KEY_LEFT_SHIFT) else speed_normal
            if rl.IsMouseButtonDown(rl.MOUSE_BUTTON_RIGHT):
                pos = np.array(AI4Animation.Standalone.IO.GetMousePositionOnScreen())
                if self.DirectionMouseStart is None:
                    self.DirectionMouseStart = pos
                else:
                    momentum = 0.01
                    self.DirectionMouseStart *= 1 - momentum
                    self.DirectionMouseStart += pos * momentum
                right_stick = [
                    pos[0] - self.DirectionMouseStart[0],
                    self.DirectionMouseStart[1] - pos[1],
                ]
            else:
                self.DirectionMouseStart = None
                right_stick = [0, 0]

            # Handle guidance selection with Q and E keys
            if rl.IsKeyPressed(rl.KEY_Q):
                self._set_guidance(self.GuidanceStyleIndex - 1)
            if rl.IsKeyPressed(rl.KEY_E):
                self._set_guidance(self.GuidanceStyleIndex + 1)

        velocity = speed * Vector3.ClampMagnitude(
            Vector3.Create(left_stick[0], 0, -left_stick[1]), 1.0
        )

        direction = Vector3.Create(right_stick[0], 0, -right_stick[1])

        position = Vector3.Lerp(
            self.SimulationObject.GetPosition(0),
            self.Actor.GetRootPosition(),
            self.Synchronization,
        )

        # Simulation
        self.SimulationObject.Control(position, direction, velocity, Time.DeltaTime)

        speed = Vector3.Length(velocity)
        self.GuidanceControl.Positions = self.GuidanceTemplates[
            "Idle" if speed < 0.1 else self.SelectedGuidance
        ].Positions.copy()

        # Correction
        if self.Sequence is not None:
            # Trajectory
            self.RootControl.Transforms = Transform.Interpolate(
                self.SimulationObject.Transforms,
                self.Sequence.Trajectory.Transforms,
                self.TrajectoryCorrection,
            )
            for i in range(self.RootControl.SampleCount):
                target = Transform.GetPosition(self.RootControl.Transforms)[i:]
                current = self.Actor.GetRootPosition().reshape(-1, 3)
                time = self.RootControl.Timestamps[i:].reshape(-1, 1)
                self.RootControl.Velocities[i] = Tensor.Sum(
                    target - current, axis=0, keepDim=False
                ) / Tensor.Sum(time, axis=0, keepDim=False)
            self.RootControl.Velocities = Vector3.Lerp(
                self.RootControl.Velocities,
                self.Sequence.Trajectory.Velocities,
                self.TrajectoryCorrection,
            )
            # Guidance
            self.GuidanceControl.Positions = Vector3.Lerp(
                self.GuidanceControl.Positions,
                self.Sequence.SampleGuidance(0.0),
                self.GuidanceCorrection,
            )

    def Predict(self):
        inputs = FeedTensor("X", self.Model.InputDim)

        root = self.Actor.Root

        transforms = Transform.TransformationTo(self.Actor.GetTransforms(), root)
        velocities = Vector3.DirectionTo(self.Actor.GetVelocities(), root)
        inputs.Feed(Transform.GetPosition(transforms))
        inputs.Feed(Transform.GetAxisZ(transforms))
        inputs.Feed(Transform.GetAxisY(transforms))
        inputs.Feed(velocities)

        futureRootTransforms = Transform.TransformationTo(
            self.RootControl.Transforms, root
        )
        futureRootVelocities = Vector3.DirectionTo(self.RootControl.Velocities, root)

        inputs.FeedVector3(
            Transform.GetPosition(futureRootTransforms), x=True, y=False, z=True
        )
        inputs.FeedVector3(
            Transform.GetAxisZ(futureRootTransforms), x=True, y=False, z=True
        )
        inputs.FeedVector3(futureRootVelocities, x=True, y=False, z=True)

        inputs.Feed(self.GuidanceControl.Positions)

        noise = 0.0
        outputs, _, _, _ = self.Model(
            inputs.GetTensor().reshape(1, -1),
            noise=(
                0.5
                - noise / 2.0
                + noise * Tensor.ToDevice(torch.rand(1, self.Model.LatentDim))
            ),
            iterations=self.NetworkIterations,
            seed=Tensor.ToDevice(torch.zeros(1, self.Model.LatentDim)),
        )
        outputs = outputs.reshape(SEQUENCE_LENGTH, -1)
        outputs = ReadTensor("Y", Tensor.ToNumPy(outputs))

        # Generate Sequence
        futureRootVectors = outputs.ReadVector3()
        futureRootDelta = Tensor.ZerosLike(futureRootVectors)
        for i in range(1, SEQUENCE_LENGTH):
            futureRootDelta[i] = futureRootDelta[i - 1] + futureRootVectors[i]
        futureRootTransforms = Transform.TransformationFrom(
            Transform.DeltaXZ(futureRootDelta), root
        )
        futureRootVelocities = Tensor.ZerosLike(futureRootVectors)
        futureRootVelocities[..., [0, 2]] = (
            futureRootVectors[..., [0, 2]] * SEQUENCE_FPS
        )
        futureRootVelocities = Vector3.DirectionFrom(
            futureRootVelocities, futureRootTransforms
        )

        futureMotionTransforms = Transform.TransformationFrom(
            Transform.TR(
                outputs.ReadVector3(self.Actor.GetBoneCount()),
                outputs.ReadRotation3D(self.Actor.GetBoneCount()),
            ),
            futureRootTransforms.reshape(SEQUENCE_LENGTH, 1, 4, 4),
        )
        futureMotionVelocities = Vector3.DirectionFrom(
            outputs.ReadVector3(self.Actor.GetBoneCount()),
            futureRootTransforms.reshape(SEQUENCE_LENGTH, 1, 4, 4),
        )

        raw_contacts = outputs.Read(4)
        futureContacts = Utility.SmoothStep(
            raw_contacts, CONTACT_THRESHOLD, CONTACT_POWER
        )

        futureGuidances = outputs.ReadVector3(self.Actor.GetBoneCount())

        self.Previous = self.Sequence
        self.Sequence = Sequence()
        self.Previous = self.Sequence if self.Previous is None else self.Previous
        self.Sequence.Timestamps = Tensor.LinSpace(
            0.0, SEQUENCE_WINDOW, SEQUENCE_LENGTH
        )
        self.Sequence.Trajectory = RootModule.Series(
            self.ControlSeries, futureRootTransforms, futureRootVelocities
        )
        self.Sequence.Motion = MotionModule.Series(
            self.ControlSeries,
            self.Actor.GetBoneNames(),
            futureMotionTransforms,
            futureMotionVelocities,
        )
        self.Sequence.Contacts = futureContacts
        self.Sequence.Guidances = futureGuidances

    def Animate(self):
        dt = Time.DeltaTime

        requiredSpeed = (
            Vector3.Distance(
                self.Actor.GetRootPosition(), self.SimulationObject.GetPosition(0)
            )
            + self.SimulationObject.GetLength()
        ) / SEQUENCE_WINDOW
        predictedSpeed = self.Sequence.GetLength() / SEQUENCE_WINDOW
        if requiredSpeed > 0.1 and predictedSpeed > 0.1:
            ts = requiredSpeed / predictedSpeed
            sync = 1.0
        else:
            ts = 1.0
            sync = 0.0
        self.Timescale = Tensor.InterpolateDt(
            self.Timescale, ts, dt, TIMESCALE_SENSITIVITY
        )
        self.Timescale = Tensor.Clamp(self.Timescale, MIN_TIMESCALE, MAX_TIMESCALE)
        self.Synchronization = Tensor.InterpolateDt(
            self.Synchronization, sync, dt, SYNCHRONIZATION_SENSITIVITY
        )

        sdt = dt * self.Timescale

        blend = (Time.TotalTime - self.Timestamp) * PREDICTION_FPS
        root = Transform.Interpolate(
            self.Previous.SampleRoot(sdt), self.Sequence.SampleRoot(sdt), blend
        )
        positions = Vector3.Lerp(
            self.Previous.SamplePositions(sdt),
            self.Sequence.SamplePositions(sdt),
            blend,
        )
        rotations = Rotation.Interpolate(
            self.Previous.SampleRotations(sdt),
            self.Sequence.SampleRotations(sdt),
            blend,
        )
        velocities = Vector3.Lerp(
            self.Previous.SampleVelocities(sdt),
            self.Sequence.SampleVelocities(sdt),
            blend,
        )
        contacts = Tensor.Interpolate(
            self.Previous.SampleContacts(sdt), self.Sequence.SampleContacts(sdt), blend
        )

        self.Actor.Root = Transform.Interpolate(
            root, self.Actor.Root, self.Sequence.GetRootLock()
        )
        self.Actor.SetTransforms(
            Transform.TR(
                Vector3.Lerp(
                    self.Actor.GetPositions() + velocities * sdt, positions, 0.5
                ),
                rotations,
            )
        )
        self.Actor.SetVelocities(velocities)

        self.Actor.RestoreBoneLengths()
        self.Actor.RestoreBoneAlignments()

        self.LeftLegIK.Solve(
            ankleContact=contacts[0],
            ballContact=contacts[1],
            maxIterations=self.SolverIterations,
            maxAccuracy=self.SolverAccuracy,
            poleTarget=Vector3.PositionFrom(
                Vector3.Create(0.0, 0.0, 1.0),
                self.Actor.GetBone(Definitions.LeftKneeName).GetTransform(),
            ),
            poleWeight=1.0,
        )
        self.RightLegIK.Solve(
            ankleContact=contacts[2],
            ballContact=contacts[3],
            maxIterations=self.SolverIterations,
            maxAccuracy=self.SolverAccuracy,
            poleTarget=Vector3.PositionFrom(
                Vector3.Create(0.0, 0.0, 1.0),
                self.Actor.GetBone(Definitions.RightKneeName).GetTransform(),
            ),
            poleWeight=1.0,
        )

        self.Actor.SyncToScene()

        self.Previous.Timestamps -= sdt
        self.Sequence.Timestamps -= sdt

    def Standalone(self):
        self.DrawRootControl = AI4Animation.GUI.Button(
            "Root Control", 0.8, 0.35, 0.15, 0.04, state=False
        )
        self.DrawGuidanceControl = AI4Animation.GUI.Button(
            "Guidance Control", 0.8, 0.40, 0.15, 0.04, state=False
        )
        self.DrawPreviousSequence = AI4Animation.GUI.Button(
            "Previous Seq", 0.8, 0.45, 0.15, 0.04, state=False
        )
        self.DrawCurrentSequence = AI4Animation.GUI.Button(
            "Current Seq", 0.8, 0.50, 0.15, 0.04, state=False
        )

        self.GuidanceDropdown = AI4Animation.GUI.Dropdown(
            f"Guidance: {self.SelectedGuidance}",
            0.375,
            0.1,
            0.25,
            0.04,
            options=[
                (name, (lambda _idx, i=i: self._set_guidance(i)))
                for i, name in enumerate(self.GuidanceNames)
            ],
        )

    def Draw(self):
        self.SimulationObject.Draw()

        if self.DrawRootControl.Active:
            self.RootControl.Draw()
        if self.DrawGuidanceControl.Active:
            self.GuidanceControl.Draw(self.Actor)
        if self.DrawPreviousSequence.Active:
            self.Previous.Draw(self.Actor, AI4Animation.Color.RED)
        if self.DrawCurrentSequence.Active:
            self.Sequence.Draw(self.Actor, AI4Animation.Color.GREEN)

    def GUI(self):
        if AI4Animation.Standalone.IO.GamepadAvailable():
            AI4Animation.Standalone.IO.DrawController(x=0.9, y=0.9, scale=0.5)
            AI4Animation.Draw.Text(
                "Left Stick: Move\nRight Stick: Facing Direction\nL1/R1: Change Style\nLeft Shift: Sprint",
                0.65,
                0.85,
                0.025,
                AI4Animation.Color.BLACK,
            )
        else:
            AI4Animation.Standalone.IO.DrawWASDQE(x=0.75, y=0.85, scale=0.5)
            AI4Animation.Draw.Text(
                "Gamepad recommended.",
                0.8,
                0.8,
                0.025,
                AI4Animation.Color.RED,
            )
            AI4Animation.Draw.Text(
                "WASD: Move\nShift: Sprint\nQ/E: Change Style\nRight Mouse Button: Facing Direction",
                0.865,
                0.85,
                0.025,
                AI4Animation.Color.BLACK,
            )

        AI4Animation.GUI.HorizontalBar(
            0.8,
            0.05,
            0.15,
            0.04,
            self.Timescale,
            label="Timescale",
            limits=[MIN_TIMESCALE, MAX_TIMESCALE],
        )
        AI4Animation.GUI.HorizontalBar(
            0.8,
            0.10,
            0.15,
            0.04,
            self.Synchronization,
            label="Synchronization",
            limits=[0.0, 1.0],
        )
        AI4Animation.GUI.HorizontalPivot(
            0.8,
            0.15,
            0.15,
            0.04,
            0.0,
            label="Previous Sequence",
            limits=[self.Previous.Timestamps[0], self.Previous.Timestamps[-1]],
            pivotColor=AI4Animation.Color.RED,
        )
        AI4Animation.GUI.HorizontalPivot(
            0.8,
            0.20,
            0.15,
            0.04,
            0.0,
            label="Current Sequence",
            limits=[self.Sequence.Timestamps[0], self.Sequence.Timestamps[-1]],
            pivotColor=AI4Animation.Color.GREEN,
        )
        AI4Animation.GUI.HorizontalBar(
            0.8,
            0.25,
            0.15,
            0.04,
            (Time.TotalTime - self.Timestamp) * PREDICTION_FPS,
            label="Blend Ratio",
        )
        AI4Animation.GUI.BarPlot(
            0.8,
            0.30,
            0.15,
            0.04,
            Tensor.SwapAxes(self.Sequence.Contacts, 0, 1),
            label="Contacts",
        )

        self.DrawRootControl.GUI()
        self.DrawGuidanceControl.GUI()
        self.DrawPreviousSequence.GUI()
        self.DrawCurrentSequence.GUI()

        self.GuidanceDropdown.Button.Label = f"Style: {self.SelectedGuidance}"
        self.GuidanceDropdown.GUI()


if __name__ == "__main__":
    AI4Animation(Program())
