import Manifold
import torch
from ai4animation import (
    FeedTensor,
    Motion,
    MotionModule,
    ReadTensor,
    RootModule,
    Tensor,
    TimeSeries,
    Transform,
)
from tqdm import tqdm
from Trinity import v3 as Definitions


class Grounding:
    def __init__(self):
        self.Model = torch.load(
            Manifold.Download(
                "ai4animation/tree/demos/motion_tracking/v0.5.0/Grounding_150.pt",
                force_redownload=False,
            ),
            weights_only=False,
            map_location="cuda" if torch.cuda.is_available() else "cpu",
        )
        self.Model.eval()
        self.BoneNames = Definitions.FULL_BODY_NAMES
        self.Resolution = 11
        self.RootSmoothing = TimeSeries(
            start=-1.0,
            end=1.0,
            samples=self.Resolution,
        )
        self.ControlSeries = TimeSeries(start=-0.5, end=0.5, samples=self.Resolution)
        self.MotionSeries = None

    def Run(self, motion) -> Motion:
        mirrored = False
        root_module = RootModule(
            motion,
            hip=Definitions.HIPS_NAME,
            left_hip=Definitions.LEFT_HIP_NAME,
            right_hip=Definitions.RIGHT_HIP_NAME,
            left_shoulder=Definitions.LEFT_SHOULDER_NAME,
            right_shoulder=Definitions.RIGHT_SHOULDER_NAME,
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
            desc="[AI4AnimationPy:Grounding]'",
        )

        body_indices = motion.GetBoneIndices(self.BoneNames)
        left_wrist_index = motion.GetBoneIndices([Definitions.LEFT_WRIST_NAME])
        right_wrist_index = motion.GetBoneIndices([Definitions.RIGHT_WRIST_NAME])
        left_finger_indices = motion.GetBoneIndices(Definitions.LEFT_HAND_NAMES)
        right_finger_indices = motion.GetBoneIndices(Definitions.RIGHT_HAND_NAMES)
        has_finger_joints = (
            len(left_finger_indices) > 0 and left_finger_indices[0] != -1
        )
        for i in pbar:
            timestamp = i * motion.DeltaTime
            root = root_module.ComputeSeries(
                timestamp=timestamp,
                mirrored=mirrored,
                timeseries=self.ControlSeries,
                smoothing=self.RootSmoothing,
            ).Transforms[5]
            result.Frames[i][body_indices] = self._ProcessFrame(
                motion, timestamp, mirrored, root=root
            )
            if has_finger_joints:
                self._CopyFingers(
                    frame=result.Frames[i],
                    motion=motion,
                    timestamp=timestamp,
                    mirrored=mirrored,
                    wrist_idx=left_wrist_index,
                    finger_indices=left_finger_indices,
                )
                self._CopyFingers(
                    frame=result.Frames[i],
                    motion=motion,
                    timestamp=timestamp,
                    mirrored=mirrored,
                    wrist_idx=right_wrist_index,
                    finger_indices=right_finger_indices,
                )

        return result

    def _CopyFingers(
        self, frame, motion, timestamp, mirrored, wrist_idx, finger_indices
    ):
        source_wrist = motion.GetBoneTransformations(
            timestamp, wrist_idx, mirrored
        ).reshape(1, 4, 4)
        source_fingers = motion.GetBoneTransformations(
            timestamp, finger_indices, mirrored
        ).reshape(-1, 4, 4)
        target_wrist = frame[wrist_idx]
        # print(source_fingers.shape, source_wrist.shape, target_wrist.shape)
        frame[finger_indices] = Transform.TransformationFromTo(
            source_fingers, source_wrist, target_wrist
        ).reshape(-1, 4, 4)

    def _Control(self, motion, timestamp, mirrored):
        # Trajectories
        transforms = motion.GetBoneTransformations(
            self.ControlSeries.SimulateTimestamps(timestamp), self.BoneNames, mirrored
        )
        self.MotionSeries = MotionModule.Series(
            self.ControlSeries, self.BoneNames, transforms
        )

    def _ProcessFrame(self, motion, timestamp, mirrored, root):
        self._Control(motion, timestamp, mirrored)
        # Inference
        # Inputs
        inputs = FeedTensor("X", self.Model.XDim)

        transforms = Transform.TransformationTo(
            self.MotionSeries.Transforms, root.reshape(-1, 1, 4, 4)
        )

        inputs.Feed(Transform.GetPosition(transforms))
        inputs.Feed(Transform.GetAxisZ(transforms))
        inputs.Feed(Transform.GetAxisY(transforms))

        # Outputs
        outputs = self.Model(inputs.GetTensor())
        outputs = ReadTensor("Y", Tensor.ToNumPy(outputs))

        body_transforms = Transform.TransformationFrom(
            Transform.TR(
                outputs.ReadVector3((1, len(self.BoneNames))),
                outputs.ReadRotation3D((1, len(self.BoneNames))),
            ),
            root,
        )
        return body_transforms
