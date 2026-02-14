# Copyright (c) Meta Platforms, Inc. and affiliates.
import os
import sys
from pathlib import Path

import torch
from ai4animation import (
    Actor,
    AdamW,
    AI4Animation,
    ContactModule,
    CyclicScheduler,
    DataSampler,
    Dataset,
    FeedTensor,
    MLP,
    MotionEditor,
    MotionModule,
    Plotting,
    ReadTensor,
    RootModule,
    Tensor,
    TimeSeries,
    Transform,
    Utility,
    Vector3,
)

SCRIPT_DIR = Path(__file__).parent
ASSETS_PATH = str(SCRIPT_DIR.parent.parent / "_ASSETS_/AnimRig")

sys.path.append(ASSETS_PATH)
import Definitions

FRAMERATE = 30
BATCH_SIZE = 32
SMOOTHING_WINDOW = 2.0
BONES = Definitions.FULL_BODY_NAMES
# BONES = [
#     Definitions.HEAD_NAME,
#     Definitions.LEFT_WRIST_NAME,
#     Definitions.RIGHT_WRIST_NAME,
#     Definitions.HIPS_NAME,
#     Definitions.LEFT_ANKLE_NAME,
#     Definitions.RIGHT_ANKLE_NAME,
# ]
RESOLUTION = 11
INPUT_DIM = RESOLUTION * len(BONES) * 9
OUTPUT_DIM = len(Definitions.FULL_BODY_NAMES) * 9

MAX_FILES = None


class Program:
    def Start(self):
        Utility.SetSeed(23456)

        self.Dataset = Dataset(
            os.path.join(ASSETS_PATH, "Motions"),
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
                lambda x: ContactModule(
                    x,
                    [
                        (Definitions.LeftAnkleName, 0.15, 0.25),
                        (Definitions.LeftBallName, 0.1, 0.25),
                        (Definitions.RightAnkleName, 0.15, 0.25),
                        (Definitions.RightBallName, 0.1, 0.25),
                    ],
                ),
            ],
            max_files=MAX_FILES,
        )

        self.DataSampler = DataSampler(
            self.Dataset,
            framerate=FRAMERATE,
            batch_size=BATCH_SIZE,
            function=self.GetTrainingFeatures,
        )

        self.RootSmoothing = TimeSeries(
            start=-SMOOTHING_WINDOW / 2.0,
            end=SMOOTHING_WINDOW / 2.0,
            samples=RESOLUTION,
        )
        self.ControlSeries = TimeSeries(start=-0.5, end=0.5, samples=RESOLUTION)

        self.EpochCount = 150
        self.DrawInterval = 500

        self.Network = Tensor.ToDevice(
            MLP.Model(
                input_dim=INPUT_DIM, output_dim=OUTPUT_DIM, hidden_dim=2048, dropout=0.1
            )
        )

        self.Optimizer = AdamW(self.Network.parameters(), lr=1e-4, weight_decay=1e-4)
        self.Scheduler = CyclicScheduler(
            optimizer=self.Optimizer,
            batch_size=self.DataSampler.BatchSize,
            epoch_size=self.DataSampler.SampleCount,
            restart_period=10,
            t_mult=2,
            policy="cosine",
            verbose=True,
        )

        self.Plotting = AI4Animation.Standalone is not None
        if self.Plotting:
            self.LossHistory = Plotting.LossHistory(
                "Loss History", drawInterval=self.DrawInterval, yScale="log"
            )

        self.Paused = False
        self.Trainer = self.Training()

    def Standalone(self):
        self.Editor = AI4Animation.Scene.AddEntity("Trainer").AddComponent(
            MotionEditor,
            self.Dataset,
            os.path.join(ASSETS_PATH, "Model.glb"),
            Definitions.FULL_BODY_NAMES,
        )
        self.Simulated = AI4Animation.Scene.AddEntity("Simulated").AddComponent(
            Actor,
            os.path.join(ASSETS_PATH, "Model.glb"),
            Definitions.FULL_BODY_NAMES,
        )
        AI4Animation.Standalone.Camera.SetTarget(self.Editor.Actor.Entity)
        self.PauseButton = AI4Animation.GUI.Button(
            "Pause Training", 0.4, 0.90, 0.2, 0.04, False, True
        )
        self.OffsetY = AI4Animation.GUI.Slider(
            0.8, 0.10, 0.15, 0.04, 0.0, -1.0, 1.0, label="Offset Y"
        )

    def GUI(self):
        self.PauseButton.GUI()
        self.OffsetY.GUI()
        self.Paused = self.PauseButton.Active

    def Update(self):
        if self.Paused:
            return
        try:
            next(self.Trainer)
        except StopIteration:
            pass

    def Training(self):
        for epoch in range(1, self.EpochCount + 1):
            for xBatch, yBatch in self.DataSampler.SampleBatchesWithinMotions(
                epoch, self.EpochCount
            ):
                _, losses = self.Network.learn(xBatch, yBatch, epoch == 1)
                self.Optimizer.zero_grad()
                sum(losses.values()).backward()
                self.Optimizer.step()
                self.Scheduler.batch_step()
                if self.Plotting:
                    for k, v in losses.items():
                        self.LossHistory.Add((Plotting.ToNumpy(v), k))
                yield

            # Epoch Step
            self.Scheduler.step()
            if self.Plotting:
                self.LossHistory.Print()

            # Save Network
            # checkpoints_dir = os.path.join(SCRIPT_DIR, "Training")
            # Utility.MakeDirectory(checkpoints_dir)
            # checkpoint_path = os.path.join(
            #     checkpoints_dir, "Network_" + "E=" + str(epoch) + ".pt"
            # )
            # print("Saving PyTorch model to " + checkpoint_path + "")
            # torch.save(self.Network, checkpoint_path)

    def GetTrainingFeatures(self, batch):
        motion, timestamps = batch
        mirrored = Tensor.RandomBool()

        rand_noise = Tensor.RandomUniform(min=0.0, max=0.1)
        rand_scale = Tensor.RandomUniform(min=0.75, max=1.25)
        rand_height = Tensor.RandomUniform(min=-0.2, max=1.0)
        ###
        ###Inputs
        ###
        inputs = FeedTensor("X", (len(timestamps), INPUT_DIM))

        root = motion.GetModule(RootModule).GetTransforms(
            timestamps, mirrored, self.RootSmoothing
        )

        yOffset = Transform.T(Vector3.Create(0.0, rand_height, 0.0))
        trajectories = motion.GetModule(MotionModule).GetTransforms(
            self.ControlSeries.SimulateTimestamps(timestamps),
            mirrored=mirrored,
            names=BONES,
            smoothing=None,
        )
        trajectories = Transform.TransformationFrom(trajectories, yOffset)

        transforms = Transform.TransformationTo(
            trajectories,
            root.reshape(-1, 1, 1, 4, 4),
        )
        positions = Transform.GetPosition(transforms)
        inputs.Feed(positions * rand_scale)
        inputs.Feed(Transform.GetAxisZ(transforms))
        inputs.Feed(Transform.GetAxisY(transforms))

        inputs = inputs.GetTensor()
        inputs += rand_noise * torch.randn_like(inputs)

        ###
        # Outputs
        ###
        outputs = FeedTensor("Y", (len(timestamps), OUTPUT_DIM))

        pose_transforms = Transform.TransformationTo(
            motion.GetBoneTransformations(
                timestamps, Definitions.FULL_BODY_NAMES, mirrored
            ),
            root.reshape(-1, 1, 4, 4),
        )

        outputs.Feed(Transform.GetPosition(pose_transforms) * rand_scale)
        outputs.Feed(Transform.GetAxisZ(pose_transforms))
        outputs.Feed(Transform.GetAxisY(pose_transforms))

        outputs = outputs.GetTensor()

        return (inputs, outputs)

    def GetEditorFeatures(self):
        features = FeedTensor("X", INPUT_DIM)
        motion = self.Editor.Motion
        timestamp = self.Editor.Timestamp
        mirrored = self.Editor.Mirror

        root = motion.GetModule(RootModule).GetTransforms(
            timestamp, mirrored, smoothing=self.RootSmoothing
        )

        offset = Transform.T(Vector3.Create(0.0, self.OffsetY.GetValue(), 0.0))

        # Trajectories
        trajectories = motion.GetModule(MotionModule).ComputeSeries(
            timestamp, mirrored, BONES, self.ControlSeries
        )

        trajectories.Transforms = Transform.TransformationFrom(
            trajectories.Transforms, offset
        )
        trajectories.Draw()

        trajectories.Transforms = Transform.TransformationTo(
            trajectories.Transforms, root.reshape(-1, 1, 4, 4)
        )

        transforms = trajectories.Transforms
        features.Feed(Transform.GetPosition(transforms))
        features.Feed(Transform.GetAxisZ(transforms))
        features.Feed(Transform.GetAxisY(transforms))

        return features.GetTensor()

    def Draw(self):
        self.Network.eval()
        with torch.no_grad():
            yPred = self.Network(self.GetEditorFeatures())
            output = ReadTensor("Y", Tensor.ToNumPy(yPred))

            motion = self.Editor.Motion
            timestamp = self.Editor.Timestamp
            mirrored = self.Editor.Mirror

            root = motion.GetModule(RootModule).GetTransforms(
                timestamp, mirrored, smoothing=self.RootSmoothing
            )

            pose = Transform.TransformationFrom(
                Transform.TR(
                    output.ReadVector3((1, len(Definitions.FULL_BODY_NAMES))),
                    output.ReadRotation3D((1, len(Definitions.FULL_BODY_NAMES))),
                ),
                root,
            )

            self.Simulated.Root = root
            self.Simulated.SetTransforms(pose)
            self.Simulated.SyncToScene()

        self.Network.train()


AI4Animation(Program(), mode=AI4Animation.Mode.STANDALONE)
