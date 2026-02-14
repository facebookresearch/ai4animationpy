# Copyright (c) Meta Platforms, Inc. and affiliates.
import datetime
import os
import sys
from functools import lru_cache
from pathlib import Path

import Manifold  # @fb-only
import torch
from ai4animation import (
    Actor,
    AI4Animation,
    Component,
    FABRIK,
    FeedTensor,
    Motion,
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
from tqdm import tqdm

SCRIPT_DIR = Path(__file__).parent
ASSETS_PATH = str(SCRIPT_DIR.parent / "_ASSETS_/Trinity3")

sys.path.append(ASSETS_PATH)
import Definitions

try:
    from .Grounding import Grounding
    from .LegIK import LegIK
    from .Sequence import Sequence

except:
    from Grounding import Grounding
    from LegIK import LegIK
    from Sequence import Sequence


class Program:
    def __init__(self, motion: Motion) -> None:
        self.SourceMotion = motion
        self.GroundedMotion = Grounding().Run(self.SourceMotion)

    def Start(self) -> None:
        self.Actor = AI4Animation.Scene.AddEntity("Actor").AddComponent(
            Actor,
            os.path.join(ASSETS_PATH, "Model.glb"),
            Definitions.FULL_BODY_NAMES_WITH_HANDS,
        )
        self.CxM = self.Actor.Entity.AddComponent(CodebookMatching, self)
        self.CxM.SetMotion(self.GroundedMotion)

    def Standalone(self):
        self.SourceActor = AI4Animation.Scene.AddEntity("Source Actor").AddComponent(
            Actor,
            os.path.join(ASSETS_PATH, "Model.glb"),
            Definitions.FULL_BODY_NAMES_WITH_HANDS,
        )
        self.GroundedActor = AI4Animation.Scene.AddEntity(
            "Grounded Actor"
        ).AddComponent(
            Actor,
            os.path.join(ASSETS_PATH, "Model.glb"),
            Definitions.FULL_BODY_NAMES_WITH_HANDS,
        )
        self.SourceActor.SkinnedMesh.SetColor(AI4Animation.Color.RED)
        self.GroundedActor.SkinnedMesh.SetColor(AI4Animation.Color.SKYBLUE)

        AI4Animation.Standalone.Camera.SetTarget(self.GroundedActor.Entity)

        source_motion = self.SourceMotion
        self.SourceMotion = Motion(
            name=source_motion.Name,
            hierarchy=source_motion.Hierarchy,
            frames=Transform.TransformationFrom(
                source_motion.Frames.copy(), Transform.T(Vector3.Create(-2, 0, 0))
            ),
            framerate=source_motion.Framerate,
        )
        self.SourceRootModule = RootModule(
            self.SourceMotion,
            hip=Definitions.HIPS_NAME,
            left_hip=Definitions.LEFT_HIP_NAME,
            right_hip=Definitions.RIGHT_HIP_NAME,
            left_shoulder=Definitions.LEFT_SHOULDER_NAME,
            right_shoulder=Definitions.RIGHT_SHOULDER_NAME,
        )

        grounded_motion = self.GroundedMotion
        self.GroundedMotion = Motion(
            name=grounded_motion.Name,
            hierarchy=grounded_motion.Hierarchy,
            frames=Transform.TransformationFrom(
                grounded_motion.Frames.copy(), Transform.T(Vector3.Create(-1, 0, 0))
            ),
            framerate=grounded_motion.Framerate,
        )
        self.GroundedRootModule = RootModule(
            self.GroundedMotion,
            hip=Definitions.HIPS_NAME,
            left_hip=Definitions.LEFT_HIP_NAME,
            right_hip=Definitions.RIGHT_HIP_NAME,
            left_shoulder=Definitions.LEFT_SHOULDER_NAME,
            right_shoulder=Definitions.RIGHT_SHOULDER_NAME,
        )

    def Update(self):
        if AI4Animation.RunMode == AI4Animation.Mode.STANDALONE:
            self.AnimateActor(
                self.SourceActor,
                self.SourceMotion,
                self.SourceRootModule,
                self.CxM.GetTimestamp(),
                self.CxM.Mirror,
            )
            self.AnimateActor(
                self.GroundedActor,
                self.GroundedMotion,
                self.GroundedRootModule,
                self.CxM.GetTimestamp(),
                self.CxM.Mirror,
            )

    def RefineMotion(self, source_motion, grounding=True) -> Motion:
        if AI4Animation.RunMode != AI4Animation.Mode.MANUAL:
            print("RefineMotion: Create AI4Animation in MANUAL mode")
            return
        grounded_motion = (
            self.GroundedMotion
            if source_motion.Name == self.SourceMotion.Name
            else Grounding().Run(source_motion)
        )
        motion = grounded_motion if grounding else source_motion
        self.CxM.SetMotion(motion)

        print(
            "Refining motion "
            + motion.Name
            + ", time: "
            + str(motion.TotalTime)
            + ", "
            + str(motion.NumFrames)
            + "frames @"
            + str(motion.Framerate)
            + "Hz"
        )

        result = Motion(
            name=motion.Name,
            hierarchy=motion.Hierarchy,
            frames=motion.Frames.copy(),
            framerate=motion.Framerate,
        )

        pbar = tqdm(
            range(motion.NumFrames),
            total=motion.NumFrames,
            desc="[AI4AnimationPy]'",
        )

        for i in pbar:
            AI4Animation.Update(motion.DeltaTime)
            result.Frames[i] = self.CxM.Actor.GetTransforms(
                motion.Hierarchy.BoneNames
            ).copy()

            pbar.set_postfix(
                {
                    "Total Time": str(
                        datetime.timedelta(seconds=round(Time.TotalTime))
                    ),
                    "Delta Time": round(Time.DeltaTime, 3),
                }
            )
        return result

    def AnimateActor(self, actor, motion, rootmodule, timestamp, mirror):
        if actor is not None:
            root = rootmodule.GetTransforms(timestamps=timestamp, mirrored=mirror)
            pose = motion.GetBoneTransformations(
                timestamp, actor.GetBoneNames(), mirror
            )
            actor.SetRoot(root)
            actor.SetTransforms(pose, actor.GetBoneNames())
            actor.SyncToScene()


@lru_cache(maxsize=5)
def _LoadModel(
    local_path: str,
    manifold_path: str,
    force_redownload: bool = False,
    weights_only: bool = False,
):
    model_path = local_path  # @oss-only
    model_path = Manifold.Download(  # @fb-only
        manifold_path,  # @fb-only
        force_redownload=force_redownload,  # @fb-only
    )  # @fb-only

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = torch.load(model_path, weights_only=weights_only, map_location=device)

    print(f"[{model_path}] loaded ...")
    model.eval()
    return model


class CodebookMatching(Component):
    def Start(self, params):
        self.Program = params[0]
        self.Actor = self.Program.Actor

        self.LowerBodyModel = _LoadModel(
            SCRIPT_DIR / "LowerbodyNetwork.pt",
            "ai4animation/tree/demos/motion_tracking/v0.5.0/Lowerbody_17.pt",
            force_redownload=False,
            weights_only=False,
        )
        self.LowerBodyModelIterations = 3

        self.UpperBodyModel = _LoadModel(
            SCRIPT_DIR / "UpperbodyNetwork.pt",
            "ai4animation/tree/demos/motion_tracking/v0.5.0/Upperbody_30.pt",
            force_redownload=False,
            weights_only=False,
        )

        self.HandModel = _LoadModel(
            SCRIPT_DIR / "HandNetwork.pt",
            "ai4animation/tree/demos/motion_tracking/v0.5.0/Hand_70.pt",
            force_redownload=False,
            weights_only=False,
        )

        self.PredictionFPS = 30
        self.SequenceLength = 16
        self.SequenceWindow = 0.5

        # Naming definitions
        self.SequenceBoneNames = Definitions.FULL_BODY_NAMES
        self.SequenceBones = self.Actor.GetBones(self.SequenceBoneNames)

        self.LowerBodyBoneNames = Definitions.LOWER_BODY_NAMES
        self.LowerBodyParentBoneNames = Definitions.LOWER_BODY_PARENT_NAMES
        self.LowerBodyBones = self.Actor.GetBones(self.LowerBodyBoneNames)
        self.LowerBodyBoneLengths = None

        self.UpperBodyBoneNames = Definitions.UPPER_BODY_NAMES
        self.UpperBodyParentBoneNames = Definitions.UPPER_BODY_PARENT_NAMES
        self.UpperBodyEndEffectorBoneNames = [
            Definitions.LEFT_WRIST_NAME,
            Definitions.RIGHT_WRIST_NAME,
            Definitions.HEAD_NAME,
        ]
        self.UpperBodyEndEffectorBones = self.Actor.GetBones(
            self.UpperBodyEndEffectorBoneNames
        )
        self.UpperBodyBones = self.Actor.GetBones(self.UpperBodyBoneNames)
        self.UpperBodyBoneLengths = None

        self.LeftWristName = Definitions.LEFT_WRIST_NAME
        self.LeftHandBoneNames = Definitions.LEFT_HAND_NAMES
        self.LeftHandParentNames = Definitions.LEFT_HAND_PARENT_NAMES
        self.LeftHandBones = self.Actor.GetBones(self.LeftHandBoneNames)

        self.RightWristName = Definitions.RIGHT_WRIST_NAME
        self.RightHandBoneNames = Definitions.RIGHT_HAND_NAMES
        self.RightHandParentNames = Definitions.RIGHT_HAND_PARENT_NAMES
        self.RightHandBones = self.Actor.GetBones(self.RightHandBoneNames)

        # Runtime Variables
        self.SmoothingSeries = TimeSeries(start=-1, end=1, samples=self.SequenceLength)
        self.SequenceSeries = TimeSeries(0.0, self.SequenceWindow, self.SequenceLength)
        self.ControlSeries = TimeSeries(0.1, 0.5, 5)
        self.MotionSeries = None
        self.RootSeries = None
        self.Sequence = None

        # PostProcessing Settings
        self.Mirror = True
        self.EulerIntegration = True
        self.RestoreAlignment = True
        self.RestoreBoneLengths = False
        self.SolveFootIK = True
        self.Contacts = [0, 0, 0, 0]  # LF Ankle, LF Ball, RF Ankle, RF Ball
        self.SolverIterations = 5
        self.SolverAccuracy = 1e-3
        self.ContactReleaseWeight = 0.5

        self.LeftLegIK: LegIK = LegIK(
            FABRIK(
                self.Actor.GetBone(Definitions.LEFT_HIP_NAME),
                self.Actor.GetBone(Definitions.LEFT_ANKLE_NAME),
            ),
            FABRIK(
                self.Actor.GetBone(Definitions.LEFT_ANKLE_NAME),
                self.Actor.GetBone(Definitions.LEFT_BALL_NAME),
            ),
        )

        self.RightLegIK: LegIK = LegIK(
            FABRIK(
                self.Actor.GetBone(Definitions.RIGHT_HIP_NAME),
                self.Actor.GetBone(Definitions.RIGHT_ANKLE_NAME),
            ),
            FABRIK(
                self.Actor.GetBone(Definitions.RIGHT_ANKLE_NAME),
                self.Actor.GetBone(Definitions.RIGHT_BALL_NAME),
            ),
        )
        self.SequencePredictionTimestamp = Time.TotalTime

    def SetMotion(self, motion):
        Time.TotalTime = 0.0
        self.SequencePredictionTimestamp = Time.TotalTime

        self.Motion = motion
        self.RootModule = RootModule(
            self.Motion,
            hip=Definitions.HIPS_NAME,
            left_hip=Definitions.LEFT_HIP_NAME,
            right_hip=Definitions.RIGHT_HIP_NAME,
            left_shoulder=Definitions.LEFT_SHOULDER_NAME,
            right_shoulder=Definitions.RIGHT_SHOULDER_NAME,
        )
        self.MotionModule = MotionModule(self.Motion)

    def GetTimestamp(self):
        return Utility.Clamp(Time.TotalTime, 0, self.Motion.TotalTime)
        # return Time.TotalTime % self.Motion.TotalTime

    def Update(self):
        if self.GetTimestamp() <= 1 / 30:
            self.InitializeTracking(self.Motion, self.GetTimestamp(), self.Mirror)

        # Predict future sequence every few frames
        if (
            self.SequencePredictionTimestamp == 0.0
            or Time.TotalTime - self.SequencePredictionTimestamp
            > 1.0 / self.PredictionFPS
        ):
            self.Control()
            self.PredictSequence()
            self.SequencePredictionTimestamp = Time.TotalTime

        # Animate motion every frame
        self.Animate(Time.DeltaTime)

    def Control(self):
        self.RootSeries = self.RootModule.ComputeSeries(
            timestamp=self.GetTimestamp(),
            mirrored=self.Mirror,
            timeseries=self.ControlSeries,
            smoothing=self.SmoothingSeries,
        )
        self.MotionSeries = self.MotionModule.ComputeSeries(
            timestamp=self.GetTimestamp(),
            mirrored=self.Mirror,
            names=self.Actor.GetBoneNames(),
            timeseries=self.ControlSeries,
            smoothing=None,
        )

    def Animate(self, deltaTime):
        if self.Sequence is None:
            return

        # Sample prediction
        root = self.Sequence.SampleRoot(deltaTime)
        positions = self.Sequence.SamplePositions(deltaTime)
        rotations = self.Sequence.SampleRotations(deltaTime)
        velocities = self.Sequence.SampleVelocities(deltaTime)
        self.Contacts = self.Sequence.SampleContacts(deltaTime)

        if self.EulerIntegration:
            positions = Vector3.Lerp(
                self.Actor.GetPositions(self.SequenceBones) + velocities * deltaTime,
                positions,
                0.5,
            )

        def GenerateLowerBody():
            self.Actor.SetTransforms(
                Transform.TR(
                    positions[: len(self.LowerBodyBones)],
                    rotations[: len(self.LowerBodyBones)],
                ),
                bones=self.LowerBodyBones,
            )
            self.Actor.SetVelocities(
                velocities[: len(self.LowerBodyBones)], bones=self.LowerBodyBones
            )

            if self.RestoreBoneLengths:
                self.Actor.SetBoneLengths(
                    self.LowerBodyBoneLengths, self.LowerBodyBones
                )

            if self.RestoreAlignment:
                self.Actor.RestoreBoneAlignments(self.LowerBodyBones)

            if self.SolveFootIK:
                self.LeftLegIK.Solve(
                    ankleContact=float(self.Contacts[0]),
                    ballContact=float(self.Contacts[1]),
                    maxIterations=self.SolverIterations,
                    maxAccuracy=self.SolverAccuracy,
                )
                self.RightLegIK.Solve(
                    ankleContact=float(self.Contacts[2]),
                    ballContact=float(self.Contacts[3]),
                    maxIterations=self.SolverIterations,
                    maxAccuracy=self.SolverAccuracy,
                )

                # Adjust velocity to fight overestimated locking
                dt = 1.0 / 30.0
                timestamp = 0.0
                boneNames = [
                    Definitions.LEFT_ANKLE_NAME,
                    Definitions.LEFT_BALL_NAME,
                    Definitions.RIGHT_ANKLE_NAME,
                    Definitions.RIGHT_BALL_NAME,
                ]
                actor_boneIndices = self.Actor.GetBoneIndices(boneNames)
                sequence_boneIndices = Tensor.Create(
                    [self.Sequence.BoneNameToIndex[name] for name in boneNames]
                ).astype(int)
                actor_root = self.Actor.GetRoot()
                count = 0
                adjusted_vel = Tensor.Zeros((len(boneNames), 3))

                while timestamp + dt <= self.Sequence.Timestamps[-1]:
                    timestamp += dt
                    sampled_root = self.Sequence.SampleRoot(timestamp).reshape(1, 4, 4)
                    sampled_positions = self.Sequence.SamplePositions(timestamp)
                    sampled_contacts = Tensor.Clamp(
                        self.Sequence.SampleContacts(timestamp), 0.0, 1.0
                    )

                    origins = Vector3.PositionTo(
                        self.Actor.GetPositions(actor_boneIndices), actor_root
                    )
                    targets = Vector3.PositionTo(
                        sampled_positions[sequence_boneIndices], sampled_root
                    )
                    momentum = (targets - origins) / timestamp
                    adjusted_vel += sampled_contacts.reshape(-1, 1) * momentum
                    count += 1

                if count > 0:
                    global_velocities = (
                        self.ContactReleaseWeight / count
                    ) * Vector3.DirectionFrom(adjusted_vel, actor_root)
                    current_velocities = self.Actor.GetVelocities(boneNames)
                    self.Actor.SetVelocities(
                        current_velocities + global_velocities, boneNames
                    )

        def GenerateUpperBody():
            bones = self.UpperBodyBones[1:].copy()
            self.Actor.SetTransforms(
                Transform.TR(
                    positions[len(self.LowerBodyBones) :],
                    rotations[len(self.LowerBodyBones) :],
                ),
                bones=bones,
            )
            self.Actor.SetVelocities(
                velocities[len(self.LowerBodyBones) :], bones=bones
            )

            if self.RestoreBoneLengths:
                self.Actor.SetBoneLengths(
                    self.UpperBodyBoneLengths, self.UpperBodyBones
                )

            if self.RestoreAlignment:
                self.Actor.RestoreBoneAlignments(bones)

        self.Actor.SetRoot(root)
        GenerateLowerBody()
        GenerateUpperBody()

        # FK on Entity system
        self.Actor.SyncToScene(self.SequenceBones)

        handMotionSeries = self.MotionModule.ComputeSeries(
            timestamp=self.GetTimestamp(),
            mirrored=self.Mirror,
            names=self.LeftHandBoneNames + self.RightHandBoneNames,
            timeseries=TimeSeries(-0.5, 0.5, 11),
            smoothing=None,
        )

        self.GenerateHand(
            wrist=self.MotionModule.GetTransforms(
                self.GetTimestamp(),
                mirrored=self.Mirror,
                names=[self.LeftWristName],
                smoothing=None,
            ),
            finger_trajectories=handMotionSeries.GetTransforms(self.LeftHandBoneNames),
            bones=self.LeftHandBones,
        )

        self.GenerateHand(
            wrist=self.MotionModule.GetTransforms(
                self.GetTimestamp(),
                mirrored=self.Mirror,
                names=[self.RightWristName],
                smoothing=None,
            ),
            finger_trajectories=handMotionSeries.GetTransforms(self.RightHandBoneNames),
            bones=self.RightHandBones,
        )

        # Update Timestamps
        self.Sequence.Timestamps -= deltaTime

    def PredictSequence(self):
        _sequence_roots = Tensor.Zeros((self.SequenceLength, 4, 4))
        _sequence_positions = Tensor.Zeros(
            (self.SequenceLength, len(self.SequenceBoneNames), 3)
        )
        _sequence_rotations = Tensor.Zeros(
            (self.SequenceLength, len(self.SequenceBoneNames), 3, 3)
        )
        _sequence_velocities = Tensor.Zeros(
            (self.SequenceLength, len(self.SequenceBoneNames), 3)
        )
        _sequence_contacts = Tensor.Zeros((self.SequenceLength, 4))

        def LowerBodyInference():
            root = self.Actor.Root.copy()
            transforms = Transform.TransformationTo(
                self.Actor.GetTransforms(self.LowerBodyBones), root
            )
            velocities = Vector3.DirectionTo(
                self.Actor.GetVelocities(self.LowerBodyBones), root
            )

            # Feed Inputs
            inputs = FeedTensor("X", self.LowerBodyModel.InputDim)

            # State
            inputs.Feed(Transform.GetPosition(transforms))
            inputs.Feed(velocities)

            # Future Root
            futureRootTransforms = Transform.TransformationTo(
                self.RootSeries.Transforms, root
            )
            inputs.FeedVector3(
                Transform.GetPosition(futureRootTransforms), x=True, y=False, z=True
            )
            inputs.FeedVector3(
                Transform.GetAxisZ(futureRootTransforms), x=True, y=False, z=True
            )

            # Future Keypoints
            futureMotionPositions = Vector3.PositionTo(
                self.MotionSeries.GetPositions(bone_names=self.LowerBodyBoneNames), root
            )
            inputs.FeedVector3(futureMotionPositions)

            # Run Network
            noise = 0.5 * Tensor.ToDevice(torch.ones(1, self.LowerBodyModel.LatentDim))
            outputs, _, _, _ = self.LowerBodyModel(
                inputs.GetTensor().reshape(1, -1),
                noise=noise,
                iterations=self.LowerBodyModelIterations,
                seed=Tensor.ToDevice(torch.zeros(1, self.LowerBodyModel.LatentDim)),
            )
            outputs = outputs.reshape(self.SequenceLength, -1)
            outputs = ReadTensor("Y", Tensor.ToNumPy(outputs))

            # Read Outputs
            futureRootVectors = outputs.ReadVector3()
            futureRootDelta = Tensor.ZerosLike(futureRootVectors)
            for i in range(1, self.SequenceLength):
                futureRootDelta[i] = futureRootDelta[i - 1] + futureRootVectors[i]
            futureRoot = Transform.TransformationFrom(
                Transform.DeltaXZ(futureRootDelta), root
            )
            futureMotionTransforms = Transform.TransformationFrom(
                Transform.TR(
                    outputs.ReadVector3(len(self.LowerBodyBones)),
                    outputs.ReadRotation3D(len(self.LowerBodyBones)),
                ),
                futureRoot.reshape(self.SequenceLength, 1, 4, 4),
            )
            futureMotionVelocities = Vector3.DirectionFrom(
                outputs.ReadVector3(len(self.LowerBodyBones)),
                futureRoot.reshape(self.SequenceLength, 1, 4, 4),
            )

            futureContacts = Tensor.Clamp(outputs.Read(4), 0, 1)

            # Current state
            _sequence_roots[0] = self.Actor.Root.copy()
            _sequence_positions[0, : len(self.LowerBodyBones)] = (
                self.Actor.GetPositions(self.LowerBodyBones).copy()
            )
            _sequence_rotations[0, : len(self.LowerBodyBones)] = (
                self.Actor.GetRotations(self.LowerBodyBones).copy()
            )
            _sequence_velocities[0, : len(self.LowerBodyBones)] = (
                self.Actor.GetVelocities(self.LowerBodyBones).copy()
            )
            _sequence_contacts[0] = futureContacts[0]

            # future states
            _sequence_roots[1:] = futureRoot[1:]
            _sequence_positions[1:, : len(self.LowerBodyBones)] = Transform.GetPosition(
                futureMotionTransforms[1:]
            )
            _sequence_rotations[1:, : len(self.LowerBodyBones)] = Transform.GetRotation(
                futureMotionTransforms[1:]
            )
            _sequence_velocities[1:, : len(self.LowerBodyBones)] = (
                futureMotionVelocities[1:]
            )
            _sequence_contacts[1:] = futureContacts[1:]

        def UpperBodyInference():
            state_transforms = self.Actor.GetTransforms(self.UpperBodyBones).copy()
            ee_state_transforms = self.Actor.GetTransforms(
                self.UpperBodyEndEffectorBones
            ).copy()
            state_velocities = self.Actor.GetVelocities(self.UpperBodyBones).copy()
            reference = Transform.TR(
                _sequence_positions[0, 0], Transform.GetRotation(_sequence_roots[0])
            )

            # Feed Inputs
            inputs = FeedTensor("X", self.UpperBodyModel.XDim)

            # State
            inputs.Feed(
                Transform.GetPosition(
                    Transform.TransformationTo(state_transforms, reference)
                )
            )
            inputs.Feed(Vector3.DirectionTo(state_velocities, reference))
            ee_state_transforms_local = Transform.TransformationTo(
                ee_state_transforms, reference
            )
            inputs.Feed(Transform.GetAxisZ(ee_state_transforms_local))
            inputs.Feed(Transform.GetAxisY(ee_state_transforms_local))

            # Future Keypoints
            futureMotionPositions = Vector3.PositionTo(
                self.MotionSeries.GetPositions(bone_names=self.UpperBodyBoneNames),
                reference,
            )
            inputs.FeedVector3(futureMotionPositions)

            futureEETransforms = Transform.TransformationTo(
                self.MotionSeries.GetTransforms(
                    bone_names=self.UpperBodyEndEffectorBoneNames
                ),
                reference,
            )
            inputs.Feed(Transform.GetAxisZ(futureEETransforms))
            inputs.Feed(Transform.GetAxisY(futureEETransforms))

            # Run Network
            outputs = self.UpperBodyModel(inputs.GetTensor())
            outputs = ReadTensor("Y", Tensor.ToNumPy(outputs))

            # Read Sequence
            _pose_p = outputs.ReadVector3(
                (self.SequenceLength, len(self.UpperBodyBones))
            )
            _pose_r = outputs.ReadRotation3D(
                (self.SequenceLength, len(self.UpperBodyBones))
            )
            _pose_v = outputs.ReadVector3(
                (self.SequenceLength, len(self.UpperBodyBones))
            )

            # Save current state of upper-body
            _sequence_positions[0, len(self.LowerBodyBones) :] = Transform.GetPosition(
                state_transforms[1:]
            )
            _sequence_rotations[0, len(self.LowerBodyBones) :] = Transform.GetRotation(
                state_transforms[1:]
            )
            _sequence_velocities[0, len(self.LowerBodyBones) :] = state_velocities[1:]

            # Future hip positions and root rotations in existing sequence
            _references = Transform.TR(
                _sequence_positions[1:, 0], Transform.GetRotation(_sequence_roots[1:])
            ).reshape(-1, 1, 4, 4)
            # Future prediction in global space
            future_positions = Vector3.PositionFrom(_pose_p[1:], _references)
            future_rotations = Rotation.RotationFrom(_pose_r[1:], _references)
            future_velocities = Vector3.DirectionFrom(_pose_v[1:], _references)

            # Overwrite future hip
            _sequence_positions[1:, :1] = future_positions[:, :1]
            _sequence_rotations[1:, :1] = future_rotations[:, :1]
            _sequence_velocities[1:, :1] = future_velocities[:, :1]
            # Write future upper-body
            _sequence_positions[1:, len(self.LowerBodyBones) :] = future_positions[
                :, 1:
            ]
            _sequence_rotations[1:, len(self.LowerBodyBones) :] = future_rotations[
                :, 1:
            ]
            _sequence_velocities[1:, len(self.LowerBodyBones) :] = future_velocities[
                :, 1:
            ]

        # Run Inference
        LowerBodyInference()
        UpperBodyInference()

        # Save Sequence
        self.Sequence = Sequence()
        self.Sequence.Timestamps = Tensor.LinSpace(
            0.0, self.SequenceWindow, self.SequenceLength
        )
        self.Sequence.RootTrajectory = RootModule.Series(
            self.SequenceSeries, _sequence_roots, None
        )
        self.Sequence.Motion = MotionModule.Series(
            self.SequenceSeries,
            self.SequenceBoneNames,
            Transform.TR(_sequence_positions, _sequence_rotations),
            _sequence_velocities,
        )
        self.Sequence.Contacts = _sequence_contacts
        self.Sequence.BoneNameToIndex = {
            name: i for i, name in enumerate(self.SequenceBoneNames)
        }

    def GenerateHand(self, wrist, finger_trajectories, bones):
        # Feed Inputs
        inputs = FeedTensor("X", self.HandModel.XDim)
        trajectories = Transform.TransformationTo(finger_trajectories, wrist)
        inputs.Feed(Transform.GetPosition(trajectories))
        inputs.Feed(Transform.GetAxisZ(trajectories))
        inputs.Feed(Transform.GetAxisY(trajectories))

        # Run Network
        outputs = self.HandModel(inputs.GetTensor())
        outputs = ReadTensor("Y", Tensor.ToNumPy(outputs))

        local_rotations = outputs.ReadRotation3D(len(bones))
        for i in range(len(bones)):
            global_rotation = Rotation.RotationFrom(
                local_rotations[i], bones[i].Parent.Entity.GetTransform()
            )
            bones[i].Entity.SetRotation(global_rotation)  # FK

        # Sync back and forth for postprocessing
        self.Actor.SyncFromScene(bones=bones, root=False)
        if self.RestoreAlignment:
            self.Actor.RestoreBoneAlignments(bones)
        self.Actor.SyncToScene(bones=bones, root=False)

    def InitializeTracking(self, motion, timestamp, mirror):
        root = self.RootModule.GetTransforms(
            timestamps=timestamp, mirrored=mirror, smoothing=None
        )
        pose = motion.GetBoneTransformations(
            timestamp, self.Actor.GetBoneNames(), mirror
        )
        velocities = motion.GetBoneVelocities(
            timestamp, self.Actor.GetBoneNames(), mirror
        )
        self.Actor.SetRoot(root)
        self.Actor.SetTransforms(pose, self.Actor.GetBoneNames())
        self.Actor.SetVelocities(velocities, self.Actor.GetBoneNames())
        self.Actor.SyncToScene()

        self.LeftLegIK.ResetTargets()
        self.RightLegIK.ResetTargets()

        _, self.LowerBodyBoneLengths = motion.GetAveragedBoneLengths(
            bone_names_or_indices=self.LowerBodyBoneNames,
            parent_names_or_indices=self.LowerBodyParentBoneNames,
            mirrored=mirror,
        )
        _, self.UpperBodyBoneLengths = motion.GetAveragedBoneLengths(
            bone_names_or_indices=self.UpperBodyBoneNames,
            parent_names_or_indices=self.UpperBodyParentBoneNames,
            mirrored=mirror,
        )
        self.LowerBodyBoneLengths = Tensor.Create(self.LowerBodyBoneLengths)
        self.UpperBodyBoneLengths = Tensor.Create(self.UpperBodyBoneLengths)

    def Draw(self):
        if self.Sequence and self.Button_Sequence.Active:
            self.Sequence.Draw(self.Actor, self.SequenceBones)
        if self.MotionSeries and self.Button_Control.Active:
            self.MotionSeries.Draw()
        if self.RootSeries and self.Button_Control.Active:
            self.RootSeries.Draw()
        if self.Button_Contacts.Active:
            AI4Animation.Draw.Sphere(
                position=self.Actor.GetBone(Definitions.LEFT_ANKLE_NAME).GetPosition(),
                size=0.065,
                color=Utility.Opacity(AI4Animation.Color.GREEN, self.Contacts[0]),
            )
            AI4Animation.Draw.Sphere(
                position=self.Actor.GetBone(Definitions.LEFT_BALL_NAME).GetPosition(),
                size=0.065,
                color=Utility.Opacity(AI4Animation.Color.SKYBLUE, self.Contacts[1]),
            )
            AI4Animation.Draw.Sphere(
                position=self.Actor.GetBone(Definitions.RIGHT_ANKLE_NAME).GetPosition(),
                size=0.065,
                color=Utility.Opacity(AI4Animation.Color.GREEN, self.Contacts[2]),
            )
            AI4Animation.Draw.Sphere(
                position=self.Actor.GetBone(Definitions.RIGHT_BALL_NAME).GetPosition(),
                size=0.065,
                color=Utility.Opacity(AI4Animation.Color.SKYBLUE, self.Contacts[3]),
            )

    def Standalone(self):
        self.Canvas = AI4Animation.GUI.Canvas(
            self.Entity.Name + "Settings", 0.75, 0.55, 0.225, 0.425
        )
        self.Button_Sequence = AI4Animation.GUI.Button(
            "Draw Sequence", 0.05, 0.1, 0.9, 0.08, False, True, self.Canvas
        )
        self.Button_Control = AI4Animation.GUI.Button(
            "Draw Control", 0.05, 0.2, 0.9, 0.08, False, True, self.Canvas
        )
        self.Button_Contacts = AI4Animation.GUI.Button(
            "Draw Contacts", 0.05, 0.3, 0.9, 0.08, False, True, self.Canvas
        )

        self.Button_LegIK = AI4Animation.GUI.Button(
            "Solve Leg IK", 0.05, 0.45, 0.9, 0.08, self.SolveFootIK, True, self.Canvas
        )
        self.Button_EulerIntegration = AI4Animation.GUI.Button(
            "Euler Integration",
            0.05,
            0.55,
            0.9,
            0.08,
            self.EulerIntegration,
            True,
            self.Canvas,
        )
        self.Button_RestoreAlignment = AI4Animation.GUI.Button(
            "Restore Bone Alignments",
            0.05,
            0.65,
            0.9,
            0.08,
            self.RestoreAlignment,
            True,
            self.Canvas,
        )
        self.Button_RestoreBoneLengths = AI4Animation.GUI.Button(
            "Restore Bone Lengths",
            0.05,
            0.75,
            0.9,
            0.08,
            self.RestoreBoneLengths,
            True,
            self.Canvas,
        )
        self.Button_Mirror = AI4Animation.GUI.Button(
            "Mirror Motion", 0.05, 0.85, 0.9, 0.08, self.Mirror, True, self.Canvas
        )

        self.Slider_ContactReleaseWeight = AI4Animation.GUI.Slider(
            0.8, 0.05, 0.15, 0.04, 0.0, 0.0, 1.0, label="Contact Release Weight"
        )

    def GUI(self):
        AI4Animation.Draw.Text(
            f"{self.Motion.Name}, speed:{Time.Timescale}x",
            0.75,
            0.45,
            0.025,
            AI4Animation.Color.BLACK,
        )
        AI4Animation.Draw.Text(
            f"Frame: {float(self.Motion.GetFrameIndices(self.GetTimestamp())[0]) + 1}/{self.Motion.NumFrames} @{float(self.Motion.Framerate):.1f}Hz",
            0.75,
            0.5,
            0.025,
            AI4Animation.Color.BLACK,
        )

        self.Canvas.GUI()
        self.Button_Sequence.GUI()
        self.Button_Control.GUI()
        self.Button_Contacts.GUI()
        self.Button_LegIK.GUI()
        self.Button_EulerIntegration.GUI()
        self.Button_RestoreAlignment.GUI()
        self.Button_RestoreBoneLengths.GUI()
        self.Button_Mirror.GUI()
        self.Slider_ContactReleaseWeight.GUI()

        self.SolveFootIK = self.Button_LegIK.Active
        self.EulerIntegration = self.Button_EulerIntegration.Active
        self.RestoreAlignment = self.Button_RestoreAlignment.Active
        self.RestoreBoneLengths = self.Button_RestoreBoneLengths.Active
        self.Mirror = self.Button_Mirror.Active
        if self.Slider_ContactReleaseWeight.Modified:
            self.ContactReleaseWeight = self.Slider_ContactReleaseWeight.GetValue()
        else:
            self.Slider_ContactReleaseWeight.SetValue(self.ContactReleaseWeight)

        if self.Program.SourceActor:
            source_pos = (
                self.Program.SourceActor.GetBone(Definitions.HEAD_NAME).GetPosition()
                + Vector3.Create((0, 0.35, 0))
            ).reshape(1, 3)
            AI4Animation.Draw.Text3D(
                ["Source"], source_pos, 0.025, AI4Animation.Color.BLACK
            )
        if self.Program.GroundedActor:
            source_pos = (
                self.Program.GroundedActor.GetBone(Definitions.HEAD_NAME).GetPosition()
                + Vector3.Create((0, 0.35, 0))
            ).reshape(1, 3)
            AI4Animation.Draw.Text3D(
                ["Grounded"], source_pos, 0.025, AI4Animation.Color.BLACK
            )
        refined_pos = (
            self.Actor.GetBone(Definitions.HEAD_NAME).GetPosition()
            + Vector3.Create((0, 0.35, 0))
        ).reshape(1, 3)
        AI4Animation.Draw.Text3D(
            ["CxM Refined"], refined_pos, 0.025, AI4Animation.Color.BLACK
        )


def main():
    Time.Timescale = 1.0
    local_glb_path = SCRIPT_DIR / "Motion.glb"  # @oss-only
    local_glb_path = Manifold.Download(  # @fb-only
        "ai4animation/tree/assets/Trinity/Data/v3/Emotes/reels/reels000.glb"  # @fb-only
    )  # @fb-only
    motion = Motion.LoadFromGLB(
        local_glb_path, names=Definitions.FULL_BODY_NAMES_WITH_HANDS, floor=None
    )
    program = Program(motion)
    AI4Animation(program, mode=AI4Animation.Mode.STANDALONE)  # STANDALONE
    program.RefineMotion(motion, grounding=True)


if __name__ == "__main__":
    main()
