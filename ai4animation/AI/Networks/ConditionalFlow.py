# Copyright (c) Meta Platforms, Inc. and affiliates.
import torch
import torch.nn as nn
from ai4animation.AI import Modules, Stats


class ConditionalFlow(nn.Module):
    def __init__(self, input_dim, flow_dim, output_dim, dropout):
        super().__init__()
        self.XDim = input_dim
        self.FlowDim = flow_dim
        self.YDim = output_dim
        self.XStats = Stats.RunningStats(input_dim)
        self.YStats = Stats.RunningStats(output_dim)
        self.Model = Modules.LinearEncoder(
            output_dim + input_dim + 1, flow_dim, output_dim, dropout
        )
        # self.Model = modules.LinearFiLMEncoder(
        #     output_dim,
        #     flow_dim,
        #     output_dim,
        #     input_dim+1,
        #     dropout
        # )

    def forward(self, x, noise=None, steps=None):
        x = self.XStats.Normalize(x)
        x_t = noise
        if type(noise) is int and noise == 0:
            x_t = torch.zeros(x.shape[0], self.FlowDim).to(x.device)
        if type(noise) is int and noise == 1:
            x_t = torch.randn(x.shape[0], self.FlowDim).to(x.device)
        if noise is None:
            x_t = torch.zeros(x.shape[0], self.FlowDim).to(x.device)
        steps = 10 if steps is None else steps
        timestamps = torch.linspace(0, 1.0, steps + 1)
        for i in range(steps):
            x_t = self.step(
                x_t=x_t, t_start=timestamps[i], t_end=timestamps[i + 1], x=x
            )
        x_t = self.YStats.Denormalize(x_t)
        return x_t

    def learn(self, input, output):
        self.XStats.Update(input)
        self.YStats.Update(output)

        input = self.XStats.Normalize(input)
        output = self.YStats.Normalize(output)

        noise = torch.randn_like(output)
        t = torch.rand(output.shape[0], 1).to(output.device)
        x_t = (1 - t) * noise + t * output

        z = self.run(x_t, t, input)

        mse = nn.MSELoss()
        loss = {"MSE Loss": mse(z, output - noise)}

        return loss, self.YStats.Denormalize(z)

    def run(self, x_t, t, x):
        # print(x_t.shape, t.shape, x.shape)
        return self.Model(torch.cat((x_t, t, x), dim=-1))
        # return self.Model(torch.cat((x_t, t), dim=-1), x)
        # return self.Model(x_t, torch.cat((t, x), -1))

    def step(self, x_t, t_start, t_end, x):
        t_start = t_start.view(1, 1).expand(x_t.shape[0], 1)
        dt = t_end - t_start
        return x_t + dt * self.run(
            x_t + dt / 2 * self.run(x_t, t_start, x), t_start + dt / 2, x
        )
