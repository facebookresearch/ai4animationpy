import torch
import torch.nn as nn
from ai4animation.AI import Modules, Stats


class Flow(nn.Module):
    def __init__(self, flow_dim, output_dim):
        super().__init__()
        self.Stats = Stats.RunningStats(output_dim)
        self.Model = Modules.LinearEncoder(output_dim + 1, flow_dim, output_dim, 0.0)

    def forward(self, noise, steps):
        x_t = noise
        timestamps = torch.linspace(0, 1.0, steps + 1)
        for i in range(steps):
            x_t = self.step(x_t=x_t, t_start=timestamps[i], t_end=timestamps[i + 1])
        x_t = self.Stats.Denormalize(x_t)
        return x_t

    def learn(self, output):
        self.Stats.Update(output)

        output = self.Stats.Normalize(output)

        noise = torch.randn_like(output)
        t = torch.rand(output.shape[0], 1).to(output.device)
        x_t = (1 - t) * noise + t * output

        z = self.run(x_t, t)

        mse = nn.MSELoss()
        loss = {"MSE Loss": mse(z, output - noise)}

        return loss, self.Stats.Denormalize(z)

    def run(self, x_t, t):
        return self.Model(torch.cat((x_t, t), dim=-1))

    def step(self, x_t, t_start, t_end):
        t_start = t_start.view(1, 1).expand(x_t.shape[0], 1)
        dt = t_end - t_start
        return x_t + dt * self.run(
            x_t + dt / 2 * self.run(x_t, t_start), t_start + dt / 2
        )
