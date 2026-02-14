# Copyright (c) Meta Platforms, Inc. and affiliates.
import numpy as np
import torch
from ai4animation import AdamW, AI4Animation, CyclicScheduler, MLP, Plotting, Tensor


class Program:
    def Start(self):
        self.EpochCount = 150
        self.BatchSize = 32
        self.BatchCount = 10
        self.SampleCount = self.BatchSize * self.BatchCount
        self.DrawInterval = 500
        self.Network = Tensor.ToDevice(
            MLP.Model(input_dim=1, output_dim=100, hidden_dim=128, dropout=0.1)
        )
        self.Optimizer = AdamW(self.Network.parameters(), lr=1e-4, weight_decay=1e-4)
        self.Scheduler = CyclicScheduler(
            optimizer=self.Optimizer,
            batch_size=self.BatchSize,
            epoch_size=self.SampleCount,
            restart_period=10,
            t_mult=2,
            policy="cosine",
            verbose=True,
        )
        self.LossHistory = Plotting.LossHistory(
            "Loss History", drawInterval=self.DrawInterval, yScale="log"
        )
        self.Trainer = self.Training()

    def Update(self):
        try:
            next(self.Trainer)
        except StopIteration as e:
            pass

    def Training(self):
        for e in range(1, self.EpochCount + 1):
            print("Epoch", e)
            for _ in range(self.BatchCount):
                x = self.GetInput()
                y = self.GetOutput(x)
                xBatch = Tensor.ToDevice(torch.tensor(x, dtype=torch.float32))
                yBatch = Tensor.ToDevice(torch.tensor(y, dtype=torch.float32))

                _, losses = self.Network.learn(xBatch, yBatch, e == 1)
                self.Optimizer.zero_grad()
                sum(losses.values()).backward()
                self.Optimizer.step()
                self.Scheduler.batch_step()

                for k, v in losses.items():
                    self.LossHistory.Add((Plotting.ToNumpy(v), k))

                yield

            self.Scheduler.step()
            self.LossHistory.Print()

    def GetInput(self):
        x = np.random.uniform(0, 1, self.BatchSize)
        x = x.reshape(self.BatchSize, 1)
        return x

    def GetOutput(self, x):
        y = np.linspace(-1, 1, 100)
        y = np.power(y, 2)
        y = y.reshape(1, -1).repeat(self.BatchSize, axis=0)
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
