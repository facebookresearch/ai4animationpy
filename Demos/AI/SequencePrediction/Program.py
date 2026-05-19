# Copyright (c) Meta Platforms, Inc. and affiliates.
import os
import sys
from pathlib import Path

import torch
from ai4animation import (
    AI4Animation,
    CosineAnnealingOptimizer,
    DataSampler,
    Dataset,
    FeedTensor,
    MirrorModule,
    MotionEditor,
    MotionModule,
    MultiLayerPerceptron,
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
ASSETS_PATH = str(SCRIPT_DIR.parent.parent / "_ASSETS_/Cranberry")
sys.path.append(ASSETS_PATH)
import Definitions

EPOCH_COUNT = 150
BATCH_SIZE = 32
FRAMERATE = 30
DRAW_INTERVAL = 500
BONES = Definitions.FULL_BODY_NAMES
FUTURE_SAMPLES = 6
INPUT_DIM = 12 * len(BONES)
OUTPUT_DIM = FUTURE_SAMPLES * 4 + FUTURE_SAMPLES * len(BONES) * 9


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
                    Definitions.NeckName,
                ),
                lambda x: MotionModule(x),
                lambda x: MirrorModule(
                    x, Vector3.Axis.ZPositive, Vector3.Create(0, 0, 180)
                ),
            ],
        )

        self.DataSampler = DataSampler(
            self.Dataset,
            framerate=FRAMERATE,
            batch_size=BATCH_SIZE,
            function=self.GetTrainingFeatures,
        )

        self.Network = Tensor.ToDevice(
            MultiLayerPerceptron.Model(
                input_dim=INPUT_DIM,
                output_dim=OUTPUT_DIM,
                hidden_dim=1024,
            )
        )

        self.Optimizer = CosineAnnealingOptimizer(
            self.Network.parameters(),
            self.DataSampler.BatchSize,
            self.DataSampler.BatchCount,
        )

        self.LossHistory = Plotting.LossHistory(
            "Loss History",
            self.DataSampler.BatchCount,
            drawInterval=DRAW_INTERVAL,
            yScale="log",
        )

        self.FutureSeries = TimeSeries(start=0.0, end=0.5, samples=FUTURE_SAMPLES)

        self.Paused = False
        self.Trainer = self.Training()

    def Standalone(self):
        self.Editor = AI4Animation.Scene.AddEntity("Editor").AddComponent(
            MotionEditor,
            self.Dataset,
            os.path.join(ASSETS_PATH, "Model.glb"),
            BONES,
        )
        AI4Animation.Standalone.Camera.SetTarget(self.Editor.Actor.Entity)
        self.PauseButton = AI4Animation.GUI.Button(
            "Pause Training", 0.4, 0.90, 0.2, 0.04, False, True
        )

    def Update(self):
        if self.Paused:
            return
        try:
            next(self.Trainer)
        except StopIteration:
            pass

    def Training(self):
        for epoch in range(1, EPOCH_COUNT + 1):
            print("Epoch", epoch)
            for xBatch, yBatch in self.DataSampler.SampleBatchesWithinMotions(
                epoch, EPOCH_COUNT
            ):
                _, loss = self.Network.learn(xBatch, yBatch, epoch == 1)
                self.Optimizer.Update(loss)
                self.LossHistory.Add(loss)
                yield
            self.LossHistory.Print()

    def GetTrainingFeatures(self, batch):
        motion, timestamps = batch
        mirrored = Tensor.RandomBool()

        inputs = FeedTensor("X", (len(timestamps), INPUT_DIM))
        outputs = FeedTensor("Y", (len(timestamps), OUTPUT_DIM))

        window = Tensor.RandomUniform(min=0.0, max=1.0)
        smoothing = TimeSeries(-window / 2, window / 2, 10)
        rootInv = Tensor.Inverse(
            motion.GetModule(RootModule).GetTransforms(
                timestamps, mirrored=mirrored, smoothing=smoothing
            )
        )

        # Inputs
        # transforms = Transform.TransformationTo(
        transforms = Transform.TransformationFrom(
            motion.GetBoneTransformations(timestamps, BONES, mirrored=mirrored),
            rootInv.reshape(-1, 1, 4, 4),
        )
        # velocities = Vector3.DirectionTo(
        velocities = Vector3.DirectionFrom(
            motion.GetBoneVelocities(timestamps, BONES, mirrored=mirrored),
            rootInv.reshape(-1, 1, 4, 4),
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
            rootInv.reshape(-1, 1, 4, 4),
        )
        # futureMotion = Transform.TransformationTo(
        futureMotion = Transform.TransformationFrom(
            motion.GetModule(MotionModule).GetTransforms(
                self.FutureSeries.SimulateTimestamps(timestamps),
                mirrored,
                BONES,
            ),
            rootInv.reshape(-1, 1, 1, 4, 4),
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
                    output.ReadVector3(FUTURE_SAMPLES, True, False, True),
                    Rotation.Look(
                        output.ReadVector3(FUTURE_SAMPLES, True, False, True),
                        Vector3.UnitY(6),
                    ),
                ),
                root,
            )
            rootSeries = RootModule.Series(self.FutureSeries, futureRoot)
            rootSeries.Draw()

            # Motion
            futureMotion = Transform.TransformationFrom(
                Transform.TR(
                    output.ReadVector3((FUTURE_SAMPLES, len(BONES))),
                    output.ReadRotation3D((FUTURE_SAMPLES, len(BONES))),
                ),
                root,
            )
            motionSeries = MotionModule.Series(self.FutureSeries, BONES, futureMotion)
            motionSeries.Draw()

        self.Network.train()

    def GUI(self):
        self.PauseButton.GUI()
        self.Paused = self.PauseButton.Active


def main():
    AI4Animation(Program(), mode=AI4Animation.Mode.STANDALONE)


if __name__ == "__main__":
    main()
