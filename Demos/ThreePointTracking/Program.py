# Copyright (c) Meta Platforms, Inc. and affiliates.
import AnimRig
import AnimRig as Definitions
import Manifold
import torch
from ai4animation import (
    Actor,
    AI4Animation,
    AssetManager,
    ContactModule,
    Dataset,
    FABRIK,
    FeedTensor,
    MotionEditor,
    MotionModule,
    ReadTensor,
    RootModule,
    Rotation,
    Tensor,
    Time,
    TimeSeries,
    TrackingModule,
    Transform,
    Vector3,
)
from LegIK import LegIK
from Sequence import Sequence

SEQUENCE_LENGTH = 16
SEQUENCE_WINDOW = 0.5
SEQUENCE_FPS = 30
PREDICTION_FPS = 15

MIN_TIMESCALE = 1.0
MAX_TIMESCALE = 1.5
TIMESCALE_SENSITIVITY = 5

BONE_COUNT = 23

MAX_FILES = None


class Program:
    def Start(self):
        dataset = Dataset(
            # AssetManager.GetPath("AnimRig/Data/Cranberry/S01"),
            # AssetManager.GetPath("Playbacks"),
            Manifold.Download("ai4animation/tree/demos/3pt/Playbacks"),
            # Utility.DownloadFileFromManifold(
            #     "ai4animation/tree/demos/3pt/Playback_1.npz",
            #     force_redownload=True,
            # ),
            [
                lambda x: RootModule(
                    x,
                    Definitions.HipName,
                    Definitions.LeftHipName,
                    Definitions.RightHipName,
                    Definitions.LeftShoulderName,
                    Definitions.RightShoulderName,
                ),
                lambda x: MotionModule(x),
                lambda x: TrackingModule(
                    x,
                    Definitions.HeadName,
                    Definitions.LeftWristName,
                    Definitions.RightWristName,
                ),
                lambda x: ContactModule(
                    x,
                    [
                        (Definitions.LeftAnkleName, 0.1, 0.25),
                        (Definitions.LeftBallName, 0.05, 0.25),
                        (Definitions.RightAnkleName, 0.1, 0.25),
                        (Definitions.RightBallName, 0.05, 0.25),
                    ],
                ),
            ],
            max_files=MAX_FILES,
        )

        self.Editor = AI4Animation.Scene.AddEntity("Editor").AddComponent(
            MotionEditor,
            dataset,
            AssetManager.GetPath("Assets/AnimRig/Model.glb"),
            Definitions.FULL_BODY_NAMES,
        )

        self.Actor = AI4Animation.Scene.AddEntity("Actor").AddComponent(
            Actor,
            AssetManager.GetPath("Assets/AnimRig/Model.glb"),
            Definitions.FULL_BODY_NAMES,
            True,
        )
        AI4Animation.Standalone.Camera.SetTarget(self.Actor.Entity)

        self.AnticipationModel = torch.load(
            Manifold.Download(
                "ai4animation/tree/demos/3pt/Network_Anticipation_E=150.pt",
                force_redownload=False,
            ),
            weights_only=False,
        )
        self.AnticipationModel.eval()

        self.MotionModel = torch.load(
            Manifold.Download(
                "ai4animation/tree/demos/3pt/Network_Motion_E=150.pt",
                force_redownload=False,
            ),
            weights_only=False,
        )
        self.MotionModel.eval()

        self.SolverIterations = 1
        self.SolverAccuracy = 1e-3

        self.NetworkIterations = 3

        self.Timescale = 1.0

        contactBones = [
            AnimRig.LeftAnkleName,
            AnimRig.LeftBallName,
            AnimRig.RightAnkleName,
            AnimRig.RightBallName,
        ]
        trackerBones = [AnimRig.HeadName, AnimRig.LeftWristName, AnimRig.RightWristName]

        self.SequenceSeries = TimeSeries(
            start=0.0, end=SEQUENCE_WINDOW, samples=SEQUENCE_LENGTH
        )

        self.TrackingHistory = TrackingModule.Series(
            TimeSeries(-SEQUENCE_WINDOW, 0.0, SEQUENCE_LENGTH), trackerBones, None, None
        )

        self.RootControl = RootModule.Series(self.SequenceSeries)
        self.MotionControl = MotionModule.Series(
            self.SequenceSeries, Definitions.FULL_BODY_NAMES
        )

        self.Previous = None
        self.Sequence = None

        self.ContactIndices = self.Actor.GetBoneIndices(contactBones)
        self.TrackerIndices = self.Actor.GetBoneIndices(trackerBones)

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

    def Update(self):
        self.Editor.Actor.GetBone(Definitions.LeftHipName).Entity.SetScale(
            Vector3.Create(0.01, 0.01, 0.01)
        )
        self.Editor.Actor.GetBone(Definitions.RightHipName).Entity.SetScale(
            Vector3.Create(0.01, 0.01, 0.01)
        )
        # Update control every frame
        self.Control()

        self.PredictAnticipation()

        # Predict future sequence every few frames
        if (
            self.Timestamp == 0.0
            or Time.TotalTime - self.Timestamp > 1.0 / PREDICTION_FPS
        ):
            self.Timestamp = Time.TotalTime
            self.PredictMotion()

        # Animate motion every frame
        self.Animate()

    def Control(self):
        self.TrackingHistory = self.Editor.Motion.GetModule(
            TrackingModule
        ).ComputeSeries(self.Editor.Timestamp, self.Editor.Mirror, self.TrackingHistory)

    def PredictAnticipation(self):
        # Reference
        position = Transform.GetPosition(self.TrackingHistory.Transforms[-1][0]).copy()
        position[..., 1] = 0.0
        rotation = Transform.GetRotation(self.RootControl.Transforms[0])
        reference = Transform.TR(position, rotation)

        transforms = Transform.TransformationTo(
            self.TrackingHistory.Transforms, reference
        )
        velocities = Vector3.DirectionTo(self.TrackingHistory.Velocities, reference)

        inputs = FeedTensor("X", self.AnticipationModel.XDim)
        inputs.Feed(Transform.GetPosition(transforms))
        inputs.Feed(Transform.GetAxisZ(transforms))
        inputs.Feed(Transform.GetAxisY(transforms))
        inputs.Feed(velocities)

        outputs = self.AnticipationModel(inputs.GetTensor().reshape(1, -1))
        outputs = ReadTensor("Y", Tensor.ToNumPy(outputs))

        # Root
        rootTransforms = Transform.TransformationFrom(
            Tensor.Squeeze(
                Transform.TR(
                    outputs.ReadVector3(SEQUENCE_LENGTH, x=True, y=False, z=True),
                    Rotation.LookPlanar(
                        outputs.ReadVector3(SEQUENCE_LENGTH, x=True, y=False, z=True)
                    ),
                ),
                0,
            ),
            reference,
        )
        rootVelocities = Vector3.DirectionFrom(
            Tensor.Squeeze(
                outputs.ReadVector3(SEQUENCE_LENGTH, x=True, y=False, z=True), 0
            ),
            reference,
        )
        self.RootControl.Transforms = rootTransforms
        self.RootControl.Velocities = rootVelocities

        # Motion
        motionTransforms = Tensor.Squeeze(
            Transform.TR(
                outputs.ReadVector3((SEQUENCE_LENGTH, BONE_COUNT)),
                outputs.ReadRotation3D((SEQUENCE_LENGTH, BONE_COUNT)),
            ),
            0,
        )
        motionVelocities = Tensor.Squeeze(
            outputs.ReadVector3((SEQUENCE_LENGTH, BONE_COUNT)), 0
        )
        motionTransforms = Transform.TransformationFrom(
            motionTransforms, rootTransforms.reshape(-1, 1, 4, 4)
        )
        motionVelocities = Vector3.DirectionFrom(
            motionVelocities, rootTransforms.reshape(-1, 1, 4, 4)
        )
        self.MotionControl.Transforms = motionTransforms
        self.MotionControl.Velocities = motionVelocities

    def PredictMotion(self):
        inputs = FeedTensor("X", self.MotionModel.InputDim)

        # Root
        root = self.Actor.Root

        # Pose
        transforms = Transform.TransformationTo(self.Actor.GetTransforms(), root)
        velocities = Vector3.DirectionTo(self.Actor.GetVelocities(), root)
        inputs.Feed(Transform.GetPosition(transforms))
        inputs.Feed(Transform.GetAxisZ(transforms))
        inputs.Feed(Transform.GetAxisY(transforms))
        inputs.Feed(velocities)

        # Motion Control
        transforms = Transform.TransformationTo(
            self.MotionControl.GetTransforms(Definitions.THREE_POINT_NAMES), root
        )
        inputs.Feed(Transform.GetPosition(transforms))
        inputs.Feed(Transform.GetAxisZ(transforms))
        inputs.Feed(Transform.GetAxisY(transforms))

        noise = 0.0
        outputs, _, _, _ = self.MotionModel(
            inputs.GetTensor().reshape(1, -1),
            noise=(
                0.5
                - noise / 2.0
                + noise * Tensor.ToDevice(torch.rand(1, self.MotionModel.LatentDim))
            ),
            iterations=self.NetworkIterations,
            seed=Tensor.ToDevice(torch.zeros(1, self.MotionModel.LatentDim)),
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

        self.Previous = self.Sequence
        self.Sequence = Sequence()
        self.Previous = self.Sequence if self.Previous is None else self.Previous
        self.Sequence.Timestamps = Tensor.LinSpace(
            0.0, SEQUENCE_WINDOW, SEQUENCE_LENGTH
        )
        ts = TimeSeries(start=0.0, end=SEQUENCE_WINDOW, samples=SEQUENCE_LENGTH)
        self.Sequence.Trajectory = RootModule.Series(
            ts, futureRootTransforms, futureRootVelocities
        )
        self.Sequence.Motion = MotionModule.Series(
            ts,
            self.Actor.GetBoneNames(),
            futureMotionTransforms,
            futureMotionVelocities,
        )
        self.Sequence.Contacts = futureContacts

    def Animate(self):
        if self.Previous is None or self.Sequence is None:
            return

        dt = Time.DeltaTime

        requiredSpeed = (
            Vector3.Distance(
                self.Actor.GetRootPosition(), self.RootControl.GetPosition(0)
            )
            + self.RootControl.GetLength()
        ) / SEQUENCE_WINDOW
        predictedSpeed = self.Sequence.GetLength() / SEQUENCE_WINDOW
        if requiredSpeed > 0.1 and predictedSpeed > 0.1:
            ts = requiredSpeed / predictedSpeed
        else:
            ts = 1.0
        self.Timescale = Tensor.Clamp(
            Tensor.InterpolateDt(self.Timescale, ts, dt, TIMESCALE_SENSITIVITY),
            MIN_TIMESCALE,
            MAX_TIMESCALE,
        )

        sdt = self.Timescale * dt

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

        # self.Actor.Root = Transform.Interpolate(root, self.Actor.Root, self.Sequence.GetRootLock())
        self.Actor.Root = root
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
        self.DrawPreviousSequence = AI4Animation.GUI.Button(
            "Previous Seq", 0.8, 0.45, 0.15, 0.04, state=False
        )
        self.DrawCurrentSequence = AI4Animation.GUI.Button(
            "Current Seq", 0.8, 0.50, 0.15, 0.04, state=False
        )
        self.DrawTrackingHistory = AI4Animation.GUI.Button(
            "Tracking History", 0.8, 0.55, 0.15, 0.04, state=False
        )
        self.DrawAnticipation = AI4Animation.GUI.Button(
            "Anticipation", 0.8, 0.60, 0.15, 0.04, state=False
        )

    def Draw(self):
        if self.DrawPreviousSequence.Active and self.Previous is not None:
            self.Previous.Draw(self.Actor, AI4Animation.Color.RED)
        if self.DrawCurrentSequence.Active and self.Sequence is not None:
            self.Sequence.Draw(self.Actor, AI4Animation.Color.GREEN)

        if self.DrawTrackingHistory.Active:
            self.TrackingHistory.Draw()
        if self.DrawAnticipation.Active:
            self.RootControl.Draw()
            self.MotionControl.Draw()

    def GUI(self):
        if self.Previous is not None and self.Sequence is not None:
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
            self.DrawPreviousSequence.GUI()
            self.DrawCurrentSequence.GUI()
            self.DrawTrackingHistory.GUI()
            self.DrawAnticipation.GUI()

        AI4Animation.GUI.HorizontalBar(
            0.8,
            0.40,
            0.15,
            0.04,
            self.Timescale,
            label="Timescale",
            limits=[MIN_TIMESCALE, MAX_TIMESCALE],
        )


def main():
    AI4Animation(Program())


if __name__ == "__main__":
    main()
