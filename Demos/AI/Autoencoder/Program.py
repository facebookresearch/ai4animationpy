# Copyright (c) Meta Platforms, Inc. and affiliates.
import os
import sys
from pathlib import Path

import torch
from ai4animation import (
    Actor,
    AdamW,
    AI4Animation,
    Autoencoder,
    ContactModule,
    CyclicScheduler,
    DataSampler,
    Dataset,
    FeedTensor,
    MotionEditor,
    MotionModule,
    Plotting,
    ReadTensor,
    RootModule,
    Rotation,
    Tensor,
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
FEATURE_DIM = 276
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
            Autoencoder.Model(
                feature_dim=FEATURE_DIM,
                hidden_dim=HIDDEN_DIM,
                latent_dim=LATENT_DIM,
                dropout=0.1,
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

        self.Trainer = self.Training()

    def Standalone(self):
        entity = AI4Animation.Scene.AddEntity("Trainer")
        self.Editor = entity.AddComponent(
            MotionEditor,
            self.Dataset,
            os.path.join(ASSETS_PATH, "Model.glb"),
            Definitions.FULL_BODY_NAMES,
        )
        self.Actor = AI4Animation.Scene.AddEntity("Actor").AddComponent(
            Actor, os.path.join(ASSETS_PATH, "Model.glb"), Definitions.FULL_BODY_NAMES
        )
        AI4Animation.Standalone.Camera.SetTarget(self.Actor.Entity)

    def Update(self):
        try:
            next(self.Trainer)
        except StopIteration as e:
            pass

    def Training(self):
        for epoch in range(1, self.EpochCount + 1):
            for batch in self.DataSampler.SampleBatchesWithinMotions(
                epoch, self.EpochCount
            ):
                _, losses = self.Network.learn(batch, epoch == 1)
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

        inputs = FeedTensor("X", (len(timestamps), FEATURE_DIM))

        root = Tensor.Inverse(
            motion.GetModule(RootModule).GetTransforms(timestamps, mirrored=mirrored)
        )

        # Inputs
        transforms = Transform.TransformationFrom(
            motion.GetBoneTransformations(timestamps, mirrored=mirrored),
            root.reshape(-1, 1, 4, 4),
        )
        velocities = Vector3.DirectionFrom(
            motion.GetBoneVelocities(timestamps, mirrored=mirrored),
            root.reshape(-1, 1, 4, 4),
        )
        inputs.Feed(Transform.GetPosition(transforms))
        inputs.Feed(Transform.GetAxisZ(transforms))
        inputs.Feed(Transform.GetAxisY(transforms))
        inputs.Feed(velocities)

        return inputs.GetTensor()

    def GetEditorFeatures(self):
        features = FeedTensor("X", FEATURE_DIM)
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
            self.Actor.Root = self.Editor.Actor.Root
            self.Actor.SetPositions(
                Vector3.PositionFrom(output.ReadVector3(23), self.Actor.Root)
            )
            self.Actor.SetRotations(
                Rotation.RotationFrom(output.ReadRotation3D(23), self.Actor.Root)
            )
            self.Actor.SetVelocities(
                Vector3.DirectionFrom(output.ReadVector3(23), self.Actor.Root)
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
