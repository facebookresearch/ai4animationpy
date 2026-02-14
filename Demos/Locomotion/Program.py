# Copyright (c) Meta Platforms, Inc. and affiliates.
import os
import sys
from pathlib import Path

import Manifold  # @fb-only
import numpy as np
import raylib as rl
import torch
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
    Vector3,
)
from LegIK import LegIK
from Sequence import Sequence

SCRIPT_DIR = Path(__file__).parent
ASSETS_PATH = str(SCRIPT_DIR.parent / "_ASSETS_/AnimRig")

sys.path.append(ASSETS_PATH)
import Definitions as AnimRig

MIN_TIMESCALE = 1.0
MAX_TIMESCALE = 1.5
SYNCHRONIZATION_SENSITIVITY = 5
TIMESCALE_SENSITIVITY = 5
SEQUENCE_WINDOW = 0.5
SEQUENCE_LENGTH = 16
SEQUENCE_FPS = 30
PREDICTION_FPS = 10


class Program:
    def Start(self):
        self.Actor = AI4Animation.Scene.AddEntity("Actor").AddComponent(
            Actor,
            os.path.join(ASSETS_PATH, "Model.glb"),
            AnimRig.FULL_BODY_NAMES,
            True,
        )
        AI4Animation.Standalone.Camera.SetTarget(self.Actor.Entity)

        device = "cuda" if torch.cuda.is_available() else "cpu"

        local_path = SCRIPT_DIR / "Network.pt"  # @oss-only
        manifold_path = (  # @fb-only
            "ai4animation/tree/demos/locomotion/codebook_matching/Training_New_30.pt"  # @fb-only
        )  # @fb-only
        local_path = Manifold.Download(manifold_path)  # @fb-only
        print("Loading model from:", local_path)
        self.Model = torch.load(local_path, weights_only=False, map_location=device)

        self.Model.eval()

        self.SolverIterations = 1
        self.SolverAccuracy = 1e-3

        self.NetworkIterations = 1

        self.Synchronization = 0.0
        self.Timescale = 1.0

        self.TrajectoryCorrection = 0.0
        self.GuidanceCorrection = 0.0

        self.ControlSeries = TimeSeries(0.0, SEQUENCE_WINDOW, SEQUENCE_LENGTH)
        self.SimulationObject = RootModule.Series(self.ControlSeries)

        self.RootControl = RootModule.Series(self.ControlSeries)
        self.GuidanceControl = GuidanceModule.Guidance(
            "Guidance", self.Actor.GetBoneNames(), self.Actor.GetPositions().copy()
        )
        self.GuidanceTemplates = {}
        directory = "Guidances"
        for path in os.listdir(directory):
            with np.load(directory + "/" + path, allow_pickle=True) as data:
                id = Path(path).stem
                names = data["Names"]
                positions = data["Positions"]
                self.GuidanceTemplates[id] = GuidanceModule.Guidance(
                    id, names, positions
                )
                print("Added Guidance:", id)

        self.Previous = None
        self.Sequence = None

        self.ContactBones = [
            AnimRig.LeftAnkleName,
            AnimRig.LeftBallName,
            AnimRig.RightAnkleName,
            AnimRig.RightBallName,
        ]
        self.ContactIndices = self.Actor.GetBoneIndices(self.ContactBones)

        self.LeftLegIK = LegIK(
            FABRIK(
                self.Actor.GetBone(AnimRig.LeftHipName),
                self.Actor.GetBone(AnimRig.LeftAnkleName),
            ),
            FABRIK(
                self.Actor.GetBone(AnimRig.LeftAnkleName),
                self.Actor.GetBone(AnimRig.LeftBallName),
            ),
        )

        self.RightLegIK: LegIK = LegIK(
            FABRIK(
                self.Actor.GetBone(AnimRig.RightHipName),
                self.Actor.GetBone(AnimRig.RightAnkleName),
            ),
            FABRIK(
                self.Actor.GetBone(AnimRig.RightAnkleName),
                self.Actor.GetBone(AnimRig.RightBallName),
            ),
        )

        self.Timestamp = Time.TotalTime

        AI4Animation.Standalone.IO.LogErrorIfGamepadNotAvailable()
        self.DirectionMouseStart = None

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
        if AI4Animation.Standalone.IO.GamepadAvailable():
            left_stick = AI4Animation.Standalone.IO.GetLeftStick()
            right_stick = AI4Animation.Standalone.IO.GetRightStick()
            speed = 7.0 if AI4Animation.Standalone.IO.IsLeftStickPressed() else 3.5
        else:
            # Keyboard control when no gamepad is available
            # WASD for left stick (velocity)
            # Left Shift for speed
            # Right Click and move mouse for right stick (direction)
            # Detail: We use momentum on the mouse direction start point to smooth out control over using GetMouseDeltaOnScreen() which is very noisy
            left_stick_vec3 = AI4Animation.Standalone.IO.GetWASDQE()
            left_stick = [left_stick_vec3[0], left_stick_vec3[2]]
            speed = 7.0 if rl.IsKeyDown(rl.KEY_LEFT_SHIFT) else 3.5
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
        if speed < 0.5:
            self.GuidanceControl.Positions = self.GuidanceTemplates[
                "Idle"
            ].Positions.copy()
        elif speed < 1.5:
            self.GuidanceControl.Positions = self.GuidanceTemplates[
                "Walk"
            ].Positions.copy()
        elif speed < 3.0:
            self.GuidanceControl.Positions = self.GuidanceTemplates[
                "Jog"
            ].Positions.copy()
        else:
            self.GuidanceControl.Positions = self.GuidanceTemplates[
                "Sprint"
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

        futureContacts = Tensor.Pow(Tensor.Clamp(outputs.Read(4), 0, 1), 1.0)

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
        )
        self.RightLegIK.Solve(
            ankleContact=contacts[2],
            ballContact=contacts[3],
            maxIterations=self.SolverIterations,
            maxAccuracy=self.SolverAccuracy,
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


def main():
    AI4Animation(Program())


if __name__ == "__main__":
    main()

    # ai4a = AI4Animation(Program(), headless=True)
    # while True:
    #     AI4Animation.Update(0.1)
