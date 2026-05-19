# Copyright (c) Meta Platforms, Inc. and affiliates.
import numpy as np
import torch
from ai4animation import (
    AI4Animation,
    CosineAnnealingOptimizer,
    MultiLayerPerceptron,
    Plotting,
    Tensor,
)

EPOCH_COUNT = 150
BATCH_SIZE = 32
BATCH_COUNT = 100
DRAW_INTERVAL = 250


class Program:
    def Start(self):
        self.Network = Tensor.ToDevice(
            MultiLayerPerceptron.Model(
                input_dim=1,
                output_dim=100,
                hidden_dim=128,
            )
        )

        self.Optimizer = CosineAnnealingOptimizer(
            self.Network.parameters(),
            BATCH_SIZE,
            BATCH_COUNT,
        )
        self.LossHistory = Plotting.LossHistory(
            "Loss History",
            BATCH_COUNT,
            drawInterval=DRAW_INTERVAL,
            yScale="log",
        )
        self.Trainer = self.Training()

    def Update(self):
        try:
            next(self.Trainer)
        except StopIteration as e:
            pass

    def Training(self):
        for e in range(1, EPOCH_COUNT + 1):
            print("Epoch", e)
            for _ in range(BATCH_COUNT):
                x = self.GetInput()
                y = self.GetOutput(x)
                xBatch = Tensor.ToDevice(torch.tensor(x, dtype=torch.float32))
                yBatch = Tensor.ToDevice(torch.tensor(y, dtype=torch.float32))
                _, loss = self.Network.learn(xBatch, yBatch, e == 1)
                self.Optimizer.Update(loss)
                self.LossHistory.Add(loss)
                yield
            self.LossHistory.Print()

    def GetInput(self):
        x = np.random.uniform(0, 1, BATCH_SIZE)
        x = x.reshape(BATCH_SIZE, 1)
        return x

    def GetOutput(self, x):
        y = np.linspace(-1, 1, 100)
        y = np.power(y, 2)
        y = y.reshape(1, -1).repeat(BATCH_SIZE, axis=0)
        y = y * x
        return y

    def Draw(self):
        self.Network.eval()
        with torch.no_grad():
            x = self.GetInput()
            y = self.GetOutput(x)
            xBatch = Tensor.ToDevice(torch.tensor(x, dtype=torch.float32))
            yBatch = Tensor.ToDevice(torch.tensor(y, dtype=torch.float32))
            yPred = self.Network(xBatch)
            self.DrawFunction(yBatch[0], AI4Animation.Color.BLACK)
            self.DrawFunction(yPred[0], AI4Animation.Color.GREEN)
        self.Network.train()

    def DrawFunction(self, y, color):
        y = Plotting.ToNumpy(y)
        x = np.linspace(-1, 1, 100)
        x = x.reshape(-1, 1)
        y = y.reshape(-1, 1)
        z = np.zeros_like(x)
        f = np.concatenate((x, y, z), axis=-1)
        AI4Animation.Draw.Sphere(f, size=0.025, color=color)


def main():
    AI4Animation(Program(), mode=AI4Animation.Mode.STANDALONE)


if __name__ == "__main__":
    main()
