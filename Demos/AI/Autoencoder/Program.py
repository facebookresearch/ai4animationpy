# Copyright (c) Meta Platforms, Inc. and affiliates.
import os
import sys
from pathlib import Path

import torch
from ai4animation import (
    Actor,
    AI4Animation,
    Autoencoder,
    CosineAnnealingOptimizer,
    DataSampler,
    Dataset,
    FeedTensor,
    MirrorModule,
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
ASSETS_PATH = str(SCRIPT_DIR.parent.parent / "_ASSETS_/Cranberry")

sys.path.append(ASSETS_PATH)
import Definitions

EPOCH_COUNT = 150
BATCH_SIZE = 32
FRAMERATE = 30
DRAW_INTERVAL = 500
BONES = Definitions.FULL_BODY_NAMES
FEATURE_DIM = 12 * len(BONES)
HIDDEN_DIM = 512
LATENT_DIM = 256


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
            Autoencoder.Model(
                feature_dim=FEATURE_DIM,
                embedding_dim=LATENT_DIM,
                hidden_dim=HIDDEN_DIM,
            )
        )

        self.Optimizer = CosineAnnealingOptimizer(
            self.Network.parameters(),
            self.DataSampler.BatchSize,
            self.DataSampler.BatchCount,
        )

        self.LossHistory = Plotting.LossHistory(
            "Loss History",
            horizon=self.DataSampler.BatchCount,
            drawInterval=DRAW_INTERVAL,
            yScale="log",
        )

        self.Trainer = self.Training()

    def Standalone(self):
        self.Editor = AI4Animation.Scene.AddEntity("Editor").AddComponent(
            MotionEditor,
            self.Dataset,
            os.path.join(ASSETS_PATH, "Model.glb"),
            BONES,
        )
        self.Actor = AI4Animation.Scene.AddEntity("Actor").AddComponent(
            Actor, os.path.join(ASSETS_PATH, "Model.glb"), BONES
        )
        self.Actor.SkinnedMesh.SetColor(AI4Animation.Color.RED)
        AI4Animation.Standalone.Camera.SetTarget(self.Actor.Entity)

    def Update(self):
        try:
            next(self.Trainer)
        except StopIteration as e:
            pass

    def Training(self):
        for epoch in range(1, EPOCH_COUNT + 1):
            print("Epoch", epoch)
            for batch in self.DataSampler.SampleBatchesWithinMotions(
                epoch, EPOCH_COUNT
            ):
                _, loss = self.Network.learn(batch, epoch == 1)
                self.Optimizer.Update(loss)
                self.LossHistory.Add(loss)
                yield
            self.LossHistory.Print()

    def GetTrainingFeatures(self, batch):
        motion, timestamps = batch
        mirrored = Tensor.RandomBool()

        inputs = FeedTensor("X", (len(timestamps), FEATURE_DIM))

        window = Tensor.RandomUniform(min=0.0, max=1.0)
        smoothing = TimeSeries(-window / 2, window / 2, 10)
        rootInv = Tensor.Inverse(
            motion.GetModule(RootModule).GetTransforms(
                timestamps, mirrored=mirrored, smoothing=smoothing
            )
        )

        # Inputs
        transforms = Transform.TransformationFrom(
            motion.GetBoneTransformations(timestamps, BONES, mirrored=mirrored),
            rootInv.reshape(-1, 1, 4, 4),
        )
        velocities = Vector3.DirectionFrom(
            motion.GetBoneVelocities(timestamps, BONES, mirrored=mirrored),
            rootInv.reshape(-1, 1, 4, 4),
        )
        inputs.Feed(Transform.GetPosition(transforms))
        inputs.Feed(Transform.GetAxisZ(transforms))
        inputs.Feed(Transform.GetAxisY(transforms))
        inputs.Feed(velocities)

        return inputs.GetTensor()

    def GetEditorFeatures(self):
        features = FeedTensor("X", FEATURE_DIM)
        root = self.Editor.Actor.Root
        transforms = Transform.TransformationTo(
            self.Editor.Actor.GetTransforms(BONES), root
        )
        velocities = Vector3.DirectionTo(self.Editor.Actor.GetVelocities(BONES), root)
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
            self.Actor.Root = self.Editor.Actor.Root
            self.Actor.SetPositions(
                Vector3.PositionFrom(output.ReadVector3(len(BONES)), self.Actor.Root)
            )
            self.Actor.SetRotations(
                Rotation.RotationFrom(
                    output.ReadRotation3D(len(BONES)), self.Actor.Root
                )
            )
            self.Actor.SetVelocities(
                Vector3.DirectionFrom(output.ReadVector3(len(BONES)), self.Actor.Root)
            )
            for bone in self.Actor.Bones:
                bone.RestoreLength()
            self.Actor.RestoreBoneAlignments()
            self.Actor.SyncToScene()
        self.Network.train()


def main():
    AI4Animation(Program(), mode=AI4Animation.Mode.STANDALONE)


if __name__ == "__main__":
    main()
