# Copyright (c) Meta Platforms, Inc. and affiliates.
import os
import sys
from pathlib import Path

import torch
from ai4animation import (
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
    Rotation,
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
INPUT_DIM = 276
OUTPUT_DIM = 24 + 1242


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
        )

        self.DataSampler = DataSampler(
            self.Dataset,
            framerate=FRAMERATE,
            batch_size=BATCH_SIZE,
            function=self.GetTrainingFeatures,
        )

        self.EpochCount = 150
        self.DrawInterval = 500
        self.Network = Tensor.ToDevice(
            MLP.Model(
                input_dim=INPUT_DIM, output_dim=OUTPUT_DIM, hidden_dim=512, dropout=0.1
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
        self.LossHistory = Plotting.LossHistory(
            "Loss History", drawInterval=self.DrawInterval, yScale="log"
        )

        self.FutureSeries = TimeSeries(start=0.0, end=0.5, samples=6)

        self.Trainer = self.Training()

    def Standalone(self):
        self.Editor = AI4Animation.Scene.AddEntity("Trainer").AddComponent(
            MotionEditor,
            self.Dataset,
            os.path.join(ASSETS_PATH, "Model.glb"),
            Definitions.FULL_BODY_NAMES,
        )
        AI4Animation.Standalone.Camera.SetTarget(self.Editor.Actor.Entity)

    def Update(self):
        try:
            next(self.Trainer)
        except StopIteration as e:
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
                for k, v in losses.items():
                    self.LossHistory.Add((Plotting.ToNumpy(v), k))
                yield
            self.Scheduler.step()
            self.LossHistory.Print()

    def GetTrainingFeatures(self, batch):
        motion, timestamps = batch
        mirrored = Tensor.RandomBool()

        inputs = FeedTensor("X", (len(timestamps), INPUT_DIM))
        outputs = FeedTensor("Y", (len(timestamps), OUTPUT_DIM))

        # root = motion.GetModule(RootModule).GetRootTransformations(timestamps, mirrored=mirrored)
        root = Tensor.Inverse(
            motion.GetModule(RootModule).GetTransforms(timestamps, mirrored=mirrored)
        )

        # Inputs
        # transforms = Transform.TransformationTo(
        transforms = Transform.TransformationFrom(
            motion.GetBoneTransformations(timestamps, mirrored=mirrored),
            root.reshape(-1, 1, 4, 4),
        )
        # velocities = Vector3.DirectionTo(
        velocities = Vector3.DirectionFrom(
            motion.GetBoneVelocities(timestamps, mirrored=mirrored),
            root.reshape(-1, 1, 4, 4),
        )
        inputs.Feed(Transform.GetPosition(transforms))
        inputs.Feed(Transform.GetAxisZ(transforms))
        inputs.Feed(Transform.GetAxisY(transforms))
        inputs.Feed(velocities)

        # Outputs
        # futureRoot = Transform.TransformationTo(
        futureRoot = Transform.TransformationFrom(
            motion.GetModule(RootModule).GetTransforms(
                self.FutureSeries.SimulateTimestamps(timestamps), mirrored
            ),
            root.reshape(-1, 1, 4, 4),
        )
        # futureMotion = Transform.TransformationTo(
        futureMotion = Transform.TransformationFrom(
            motion.GetModule(MotionModule).GetTransforms(
                self.FutureSeries.SimulateTimestamps(timestamps),
                mirrored,
                Definitions.FULL_BODY_NAMES,
            ),
            root.reshape(-1, 1, 1, 4, 4),
        )
        outputs.FeedVector3(Transform.GetPosition(futureRoot), x=True, y=False, z=True)
        outputs.FeedVector3(Transform.GetAxisZ(futureRoot), x=True, y=False, z=True)
        outputs.Feed(Transform.GetPosition(futureMotion))
        outputs.Feed(Rotation.GetAxisZ(futureMotion))
        outputs.Feed(Rotation.GetAxisY(futureMotion))

        return (inputs.GetTensor(), outputs.GetTensor())

    def GetEditorFeatures(self):
        features = FeedTensor("X", INPUT_DIM)
        root = self.Editor.Actor.Root
        transforms = Transform.TransformationTo(self.Editor.Actor.GetTransforms(), root)
        velocities = Vector3.DirectionTo(self.Editor.Actor.GetVelocities(), root)
        features.Feed(Transform.GetPosition(transforms))
        features.Feed(Transform.GetAxisZ(transforms))
        features.Feed(Transform.GetAxisY(transforms))
        features.Feed(velocities)
        return features.GetTensor()

    def Draw(self):
        self.Network.eval()
        with torch.no_grad():
            xBatch = self.GetEditorFeatures()
            yPred = Tensor.ToNumPy(self.Network(xBatch))
            output = ReadTensor("Y", yPred)
            root = self.Editor.Actor.Root

            # Trajectory
            futureRoot = Transform.TransformationFrom(
                Transform.TR(
                    output.ReadVector3(6, True, False, True),
                    Rotation.Look(
                        output.ReadVector3(6, True, False, True), Vector3.UnitY(6)
                    ),
                ),
                root,
            )
            rootSeries = RootModule.Series(self.FutureSeries, futureRoot)
            rootSeries.Draw()

            # Motion
            futureMotion = Transform.TransformationFrom(
                Transform.TR(
                    output.ReadVector3((6, 23)), output.ReadRotation3D((6, 23))
                ),
                root,
            )
            motionSeries = MotionModule.Series(
                self.FutureSeries, Definitions.FULL_BODY_NAMES, futureMotion
            )
            motionSeries.Draw()

        self.Network.train()


def main():
    AI4Animation(Program(), mode=AI4Animation.Mode.STANDALONE)


if __name__ == "__main__":
    main()
