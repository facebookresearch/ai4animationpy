# Copyright (c) Meta Platforms, Inc. and affiliates.

# This demo is a rework of the MANN Siggraph 2018 paper "Mode adaptive Neural Networks for Quadruped Motion Control" by Starke et al.
# The codebook matching model is trained on the same data as in the paper.

import os
import sys
from pathlib import Path

import numpy as np
import raylib as rl
import torch

SCRIPT_DIR = Path(__file__).parent
ASSETS_PATH = str(SCRIPT_DIR.parent.parent / "_ASSETS_/Quadruped")

sys.path.append(ASSETS_PATH)
import Definitions
from ai4animation import (
    Actor,
    AI4Animation,
    FABRIK,
    FeedTensor,
    GuidanceModule,
    MotionModule,
    PID,
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
INPUT_DEADZONE = 0.25
ACTION_TRIGGER_SPEED_MAX = 0.5

LOCOMOTION_MODES = {
    "walk": 0.7,
    "pace": 1.2,
    "trot": 2.0,
    "canter": 4.0,
}

CHARACTER_MODELS = {
    "dog": "Dog.glb",
    "wolf": "Wolf.glb",
}


class Program:
    def __init__(self):
        self.Character = "dog"
        self.PIDHistoryLength = 48

    def CreateActor(self, character, visible):
        actor = AI4Animation.Scene.AddEntity(
            f"Actor_{character.capitalize()}"
        ).AddComponent(
            Actor,
            os.path.join(ASSETS_PATH, CHARACTER_MODELS[character]),
            Definitions.FULL_BODY_NAMES,
            False,
        )
        if not visible:
            actor.SkinnedMesh.Unregister()
        return actor

    def CopyActorState(self, source, target):
        if source is None or target is None or source is target:
            return

        common_bones = [name for name in target.GetBoneNames() if source.HasBone(name)]
        target.SetRoot(np.array(source.GetRoot(), copy=True))
        if common_bones:
            target.SetTransforms(
                np.array(source.GetTransforms(common_bones), copy=True),
                common_bones,
            )
            target.SetVelocities(
                np.array(source.GetVelocities(common_bones), copy=True),
                common_bones,
            )
        target.SyncToScene()

    def SyncInactiveActors(self):
        for name, actor in self.Actors.items():
            if actor is not self.Actor:
                self.CopyActorState(self.Actor, actor)

    def ConfigureActorBindings(self):
        self.ContactIndices = self.Actor.GetBoneIndices(self.ContactBones)

        self.LeftHandIK = LegIK(
            FABRIK(
                self.Actor.GetBone(Definitions.LeftForeArmName),
                self.Actor.GetBone(Definitions.LeftHandSiteName),
            )
        )

        self.RightHandIK = LegIK(
            FABRIK(
                self.Actor.GetBone(Definitions.RightForeArmName),
                self.Actor.GetBone(Definitions.RightHandSiteName),
            )
        )

        self.LeftFootIK = LegIK(
            FABRIK(
                self.Actor.GetBone(Definitions.LeftKneeName),
                self.Actor.GetBone(Definitions.LeftFootSiteName),
            )
        )
        self.RightFootIK = LegIK(
            FABRIK(
                self.Actor.GetBone(Definitions.RightKneeName),
                self.Actor.GetBone(Definitions.RightFootSiteName),
            )
        )

    def SwitchCharacter(self, character):
        character = character.lower()
        if character == self.Character or character not in self.Actors:
            return

        previous_actor = self.Actor
        self.Character = character
        self.CharacterModel = CHARACTER_MODELS[character]
        self.Actor = self.Actors[character]

        self.CopyActorState(previous_actor, self.Actor)
        previous_actor.SkinnedMesh.Unregister()
        self.Actor.SkinnedMesh.Register()

        AI4Animation.Standalone.Camera.SetTarget(self.Actor.Entity)
        self.ConfigureActorBindings()

    def Start(self):
        self.CharacterModel = CHARACTER_MODELS[self.Character]
        self.Actors = {
            name: self.CreateActor(name, visible=(name == self.Character))
            for name in CHARACTER_MODELS
        }
        self.Actor = self.Actors[self.Character]
        AI4Animation.Standalone.Camera.SetTarget(self.Actor.Entity)

        self.Model = torch.load(
            os.path.join(SCRIPT_DIR, "Network.pt"), weights_only=False
        )
        self.Model.eval()

        self.PostProcessor = torch.load(
            os.path.join(SCRIPT_DIR, "PostProcessor.pt"), weights_only=False
        )
        self.PostProcessor.eval()

        self.SolverIterations = 1
        self.SolverAccuracy = 1e-3

        self.NetworkIterations = 1

        self.Timescale = 1.0
        self.Synchronization = 1.0

        self.TrajectoryCorrection = 0.1
        self.ContactPower = 2.0

        self.PID = PID(kp=2.0, ki=0.03, kd=0.0)
        self.PIDSpeedHistory = np.zeros((3, self.PIDHistoryLength), dtype=np.float32)

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

        self.CurrentGuidanceState = "Sit"
        self.GuidanceControl.Positions = self.GuidanceTemplates[
            self.CurrentGuidanceState
        ].Positions.copy()

        self.Previous = None
        self.Sequence = None

        self.ContactBones = [
            Definitions.LeftHandSiteName,
            Definitions.RightHandSiteName,
            Definitions.LeftFootSiteName,
            Definitions.RightFootSiteName,
        ]
        self.ConfigureActorBindings()

        self.Timestamp = Time.TotalTime

        AI4Animation.Standalone.IO.LogErrorIfGamepadNotAvailable()

    def CameraRelativeInput(self, x, y):
        forward = (
            self.Actor.GetRootPosition()
            - AI4Animation.Standalone.Camera.Entity.GetPosition()
        )
        forward[1] = 0.0
        if Vector3.Length(forward) == 0.0:
            forward = self.Actor.GetRootDirection()
        forward = Vector3.Normalize(forward)

        right = Vector3.Cross(forward, Vector3.Y)
        right[1] = 0.0
        right = Vector3.Normalize(right)

        return x * right + y * forward

    def _apply_deadzone(self, value, threshold):
        return Vector3.Zero() if Vector3.Length(value) < threshold else value

    def _select_locomotion_mode_keyboard(self):
        if rl.IsKeyDown(rl.KEY_LEFT_ALT) or rl.IsKeyDown(rl.KEY_RIGHT_ALT):
            return "walk"
        if rl.IsKeyDown(rl.KEY_LEFT_CONTROL) or rl.IsKeyDown(rl.KEY_RIGHT_CONTROL):
            return "trot"
        if rl.IsKeyDown(rl.KEY_LEFT_SHIFT) or rl.IsKeyDown(rl.KEY_RIGHT_SHIFT):
            return "canter"
        return "pace"

    def GetCurrentSpeed(self):
        if self.Sequence is None:
            return 0.0
        return float((self.Sequence.GetLength() / SEQUENCE_WINDOW).item())

    def _UpdatePIDSpeedHistory(self, current_speed, target_speed, pid_speed):
        self.PIDSpeedHistory = np.roll(self.PIDSpeedHistory, -1, axis=1)
        self.PIDSpeedHistory[0, -1] = float(current_speed)
        self.PIDSpeedHistory[1, -1] = float(target_speed)
        self.PIDSpeedHistory[2, -1] = float(pid_speed)

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
        # Read movement + action inputs and compute desired speed.
        current_speed = self.GetCurrentSpeed()

        if AI4Animation.Standalone.IO.GamepadAvailable():
            move_axes = AI4Animation.Standalone.IO.GetLeftStick()
            move_axes_magnitude = np.clip(np.linalg.norm(move_axes), 0.0, 1.0)

            if move_axes_magnitude > INPUT_DEADZONE:
                gait_max_speed = (
                    LOCOMOTION_MODES["canter"]
                    if AI4Animation.Standalone.IO.IsLeftStickPressed()
                    else LOCOMOTION_MODES["trot"]
                )
                desired_speed = move_axes_magnitude * gait_max_speed
            else:
                desired_speed = 0.0

            sit_requested = AI4Animation.Standalone.IO.IsR1Down()
            stand_requested = AI4Animation.Standalone.IO.IsL1Down()
            lie_requested = AI4Animation.Standalone.IO.IsL2Down()
        else:
            keyboard_move = AI4Animation.Standalone.IO.GetWASDQE()

            move_axes = Vector3.Create(keyboard_move[0], 0, keyboard_move[2])
            if keyboard_move[1] != 0:
                self.RotationIntegral += 120.0 * keyboard_move[1] * Time.DeltaTime
                move_axes = Vector3.DirectionFrom(
                    move_axes, Rotation.Euler(0, self.RotationIntegral, 0)
                )
            else:
                self.RotationIntegral = 0.0
            move_axes = [move_axes[0], move_axes[2]]

            if Vector3.Length(keyboard_move) > INPUT_DEADZONE:
                desired_speed = LOCOMOTION_MODES[
                    self._select_locomotion_mode_keyboard()
                ]
            else:
                desired_speed = 0.0

            sit_requested = rl.IsKeyDown(rl.KEY_R)
            stand_requested = rl.IsKeyDown(rl.KEY_T)
            lie_requested = rl.IsKeyDown(rl.KEY_V)

        can_trigger_action_pose = current_speed < ACTION_TRIGGER_SPEED_MAX
        sit_active = can_trigger_action_pose and sit_requested
        stand_active = can_trigger_action_pose and stand_requested
        lie_active = can_trigger_action_pose and lie_requested

        action_pose_active = sit_active or lie_active or stand_active
        target_speed = 0.0 if action_pose_active else desired_speed

        # Smooth target speed with PID and clamp to valid locomotion range
        speed = current_speed + self.PID(
            current_speed, Time.DeltaTime, setpoint=target_speed
        )
        # speed = Tensor.Clamp(speed, 0.0, LOCOMOTION_MODES["canter"])
        speed = max(speed, 0.0)

        # Build movement direction from camera-relative input
        move_vector = Vector3.ClampMagnitude(
            self._apply_deadzone(
                self.CameraRelativeInput(move_axes[0], move_axes[1]),
                INPUT_DEADZONE,
            ),
            1.0,
        )
        move_vector_length = Vector3.Length(move_vector)
        move_direction = (
            Vector3.Zero()
            if move_vector_length == 0.0
            else move_vector / move_vector_length
        )

        if action_pose_active:
            speed = 0.0
            velocity = Vector3.Zero()
            direction = self.Actor.GetRootDirection()
        else:
            velocity = speed * move_direction
            direction = velocity

        self._UpdatePIDSpeedHistory(current_speed, target_speed, speed)

        position = Vector3.Lerp(
            self.SimulationObject.GetPosition(0),
            self.Actor.GetRootPosition(),
            self.Synchronization,
        )

        # Simulation
        self.SimulationObject.Control(position, direction, velocity, Time.DeltaTime)

        speed = Vector3.Length(velocity)
        if sit_active:
            guidance_state = "Sit"
        elif lie_active:
            guidance_state = "Lie"
        elif stand_active:
            guidance_state = "Stand"
        elif speed < 0.1:
            guidance_state = "Sit" if self.Sequence is None else "Idle"
        elif speed < LOCOMOTION_MODES["pace"]:
            guidance_state = "Walk"
        elif speed < LOCOMOTION_MODES["trot"]:
            guidance_state = "Pace"
        elif speed < LOCOMOTION_MODES["canter"]:
            guidance_state = "Trot"
        else:
            guidance_state = "Canter"

        self.CurrentGuidanceState = guidance_state
        self.GuidanceControl.Positions = self.GuidanceTemplates[
            guidance_state
        ].Positions.copy()

        self.RootControl.Transforms = self.SimulationObject.Transforms.copy()
        self.RootControl.Velocities = self.SimulationObject.Velocities.copy()

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

    def Predict(self):
        inputs = FeedTensor("X", self.Model.input_dim())

        root = self.Actor.Root

        transforms = Transform.TransformationTo(self.Actor.GetTransforms(), root)
        velocities = Vector3.DirectionTo(self.Actor.GetVelocities(), root)
        inputs.Feed(Transform.GetPosition(transforms))
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

        outputs = self.Model(
            inputs.GetTensor().reshape(1, -1), iterations=self.NetworkIterations
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

        # Predict Contacts
        inputs = FeedTensor("X", self.PostProcessor.input_dim())

        currentTransforms = self.Actor.GetTransforms(self.ContactIndices)
        currentVelocities = self.Actor.GetVelocities(self.ContactIndices)
        targetTransforms = self.Sequence.Motion.GetTransforms(self.ContactBones)[
            1:, :, :
        ]
        targetVelocities = self.Sequence.Motion.GetVelocities(self.ContactBones)[
            1:, :, :
        ]
        delta_distances = Vector3.Distance(
            Transform.GetPosition(currentTransforms),
            Transform.GetPosition(targetTransforms),
        )
        delta_angles = Rotation.Angle(
            Transform.GetRotation(currentTransforms),
            Transform.GetRotation(targetTransforms),
        )
        delta_velocities = Vector3.Distance(currentVelocities, targetVelocities)

        inputs.Feed(Transform.GetPosition(transforms))
        inputs.Feed(Transform.GetAxisZ(transforms))
        inputs.Feed(Transform.GetAxisY(transforms))
        inputs.Feed(velocities)
        inputs.Feed(delta_distances)
        inputs.Feed(delta_angles)
        inputs.Feed(delta_velocities)

        contacts = Tensor.ToNumPy(
            self.PostProcessor(inputs.GetTensor()).reshape(
                SEQUENCE_LENGTH, len(self.ContactBones)
            )
        )
        self.Sequence.Contacts = Tensor.Pow(
            Tensor.Clamp(contacts, 0, 1), self.ContactPower
        )

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

        self.LeftHandIK.Solve(
            contact=contacts[0],
            maxIterations=self.SolverIterations,
            maxAccuracy=self.SolverAccuracy,
        )
        self.RightHandIK.Solve(
            contact=contacts[1],
            maxIterations=self.SolverIterations,
            maxAccuracy=self.SolverAccuracy,
        )
        self.LeftFootIK.Solve(
            contact=contacts[2],
            maxIterations=self.SolverIterations,
            maxAccuracy=self.SolverAccuracy,
        )
        self.RightFootIK.Solve(
            contact=contacts[3],
            maxIterations=self.SolverIterations,
            maxAccuracy=self.SolverAccuracy,
        )

        self.Actor.SyncToScene()
        self.SyncInactiveActors()

        self.Previous.Timestamps -= sdt
        self.Sequence.Timestamps -= sdt

    def Standalone(self):
        self.CharacterCanvas = AI4Animation.GUI.Canvas(
            "Character", 0.01, 0.35, 0.125, 0.125
        )
        self.ButtonDog = AI4Animation.GUI.Button(
            "Dog",
            0.25,
            0.3,
            0.42,
            0.25,
            state=self.Character == "dog",
            toggle=False,
            canvas=self.CharacterCanvas,
        )
        self.ButtonWolf = AI4Animation.GUI.Button(
            "Wolf",
            0.25,
            0.6,
            0.42,
            0.25,
            state=self.Character == "wolf",
            toggle=False,
            canvas=self.CharacterCanvas,
        )
        self.CharacterCanvas.AddItem(self.ButtonDog)
        self.CharacterCanvas.AddItem(self.ButtonWolf)

        self.DrawRootControl = AI4Animation.GUI.Button(
            "Root Control", 0.8, 0.40, 0.15, 0.04, state=False
        )
        self.DrawGuidanceControl = AI4Animation.GUI.Button(
            "Guidance Control", 0.8, 0.45, 0.15, 0.04, state=False
        )
        self.DrawPreviousSequence = AI4Animation.GUI.Button(
            "Previous Seq", 0.8, 0.50, 0.15, 0.04, state=False
        )
        self.DrawCurrentSequence = AI4Animation.GUI.Button(
            "Current Seq", 0.8, 0.55, 0.15, 0.04, state=False
        )

        self.SliderKp = AI4Animation.GUI.Slider(
            0.8, 0.82, 0.15, 0.03, self.PID.Kp, 0.0, 5.0
        )
        self.SliderKi = AI4Animation.GUI.Slider(
            0.8, 0.87, 0.15, 0.03, self.PID.Ki, 0.0, 5.0
        )
        self.SliderKd = AI4Animation.GUI.Slider(
            0.8, 0.92, 0.15, 0.03, self.PID.Kd, 0.0, 5.0
        )

    def Draw(self):
        self.SimulationObject.Draw()

        if self.DrawRootControl.Active:
            self.RootControl.Draw()
        if self.DrawGuidanceControl.Active:
            self.GuidanceControl.DrawLegacy(self.Actor)
        if self.DrawPreviousSequence.Active:
            self.Previous.Draw(self.Actor, AI4Animation.Color.RED)
        if self.DrawCurrentSequence.Active:
            self.Sequence.Draw(self.Actor, AI4Animation.Color.GREEN)

    def GUI(self):
        self.ButtonDog.Active = self.Character == "dog"
        self.ButtonWolf.Active = self.Character == "wolf"
        self.CharacterCanvas.GUI()

        if self.ButtonDog.IsPressed():
            self.SwitchCharacter("dog")
        if self.ButtonWolf.IsPressed():
            self.SwitchCharacter("wolf")

        if AI4Animation.Standalone.IO.GamepadAvailable():
            AI4Animation.Standalone.IO.DrawController(x=0.50, y=0.90, scale=0.4)
            AI4Animation.Draw.Text(
                "Left Stick: Move\nL1: Stand\nL2: Lie\nR1: Sit\nLeft Stick Pressed: Sprint",
                0.35,
                0.88,
                0.02,
                AI4Animation.Color.BLACK,
            )
        else:
            AI4Animation.Standalone.IO.DrawWASDQE(x=0.44, y=0.90, scale=0.4)
            AI4Animation.Draw.Text(
                "Gamepad recommended.",
                0.45,
                0.85,
                0.02,
                AI4Animation.Color.RED,
            )
            AI4Animation.Draw.Text(
                "WASD: Move\nAlt: Slow\nCtrl: Medium\nShift: Fast",
                0.55,
                0.90,
                0.02,
                AI4Animation.Color.BLACK,
            )
            AI4Animation.Draw.Text(
                "R: Sit\nT: Stand\nV: Lie",
                0.62,
                0.90,
                0.02,
                AI4Animation.Color.BLACK,
            )
        AI4Animation.Draw.Text(
            f"Guidance: {self.CurrentGuidanceState}",
            0.8,
            0.01,
            0.025,
            AI4Animation.Color.BLACK,
        )
        AI4Animation.GUI.HorizontalBar(
            0.8,
            0.05,
            0.15,
            0.04,
            self.GetCurrentSpeed(),
            label=f"{self.GetCurrentSpeed():.2f} (m/s)",
            limits=[0.0, max(LOCOMOTION_MODES.values())],
        )
        AI4Animation.GUI.HorizontalBar(
            0.8,
            0.10,
            0.15,
            0.04,
            self.Timescale,
            label="Timescale",
            limits=[MIN_TIMESCALE, MAX_TIMESCALE],
        )
        AI4Animation.GUI.HorizontalBar(
            0.8,
            0.15,
            0.15,
            0.04,
            self.Synchronization,
            label="Synchronization",
            limits=[0.0, 1.0],
        )
        AI4Animation.GUI.HorizontalPivot(
            0.8,
            0.20,
            0.15,
            0.04,
            0.0,
            label="Previous Sequence",
            limits=[self.Previous.Timestamps[0], self.Previous.Timestamps[-1]],
            pivotColor=AI4Animation.Color.RED,
        )
        AI4Animation.GUI.HorizontalPivot(
            0.8,
            0.25,
            0.15,
            0.04,
            0.0,
            label="Current Sequence",
            limits=[self.Sequence.Timestamps[0], self.Sequence.Timestamps[-1]],
            pivotColor=AI4Animation.Color.GREEN,
        )
        AI4Animation.GUI.HorizontalBar(
            0.8,
            0.30,
            0.15,
            0.04,
            (Time.TotalTime - self.Timestamp) * PREDICTION_FPS,
            label="Blend Ratio",
        )
        AI4Animation.GUI.BarPlot(
            0.8,
            0.35,
            0.15,
            0.04,
            Tensor.SwapAxes(self.Sequence.Contacts, 0, 1),
            label="Contacts",
        )

        AI4Animation.GUI.CurvePlot(
            0.8,
            0.64,
            0.15,
            0.14,
            self.PIDSpeedHistory,
            label="PID",
            min=0.0,
            max=max(LOCOMOTION_MODES.values()),
            colors=[
                AI4Animation.Color.RED,
                AI4Animation.Color.BLUE,
                AI4Animation.Color.GREEN,
            ],
            curveLabels=["Current", "Target", "PID"],
            backgroundColor=Utility.Opacity(AI4Animation.Color.BLACK, 0.4),
            frameColor=AI4Animation.Color.WHITE,
        )

        # self.SliderKp.GUI()
        # self.SliderKi.GUI()
        # self.SliderKd.GUI()
        # self.PID.Kp = self.SliderKp.GetValue()
        # self.PID.Ki = self.SliderKi.GetValue()
        # self.PID.Kd = self.SliderKd.GetValue()

        # AI4Animation.Draw.Text(
        #     f"Kp: {self.PID.Kp:.2f}",
        #     0.8,
        #     0.795,
        #     0.02,
        #     AI4Animation.Color.BLACK,
        # )
        # AI4Animation.Draw.Text(
        #     f"Ki: {self.PID.Ki:.2f}",
        #     0.8,
        #     0.845,
        #     0.02,
        #     AI4Animation.Color.BLACK,
        # )
        # AI4Animation.Draw.Text(
        #     f"Kd: {self.PID.Kd:.2f}",
        #     0.8,
        #     0.895,
        #     0.02,
        #     AI4Animation.Color.BLACK,
        # )

        self.DrawRootControl.GUI()
        self.DrawGuidanceControl.GUI()
        self.DrawPreviousSequence.GUI()
        self.DrawCurrentSequence.GUI()


if __name__ == "__main__":
    AI4Animation(Program())
