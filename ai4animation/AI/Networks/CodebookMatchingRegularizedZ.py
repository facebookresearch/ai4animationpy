import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from ai4animation.AI import Manifolds, Modules, Plotting, Stats


class Encoder(nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_dim,
        latent_dim,
        code_dim,
        sequence_length,
        hard,
        dropout,
    ):
        super(Encoder, self).__init__()
        self.CodeDim = code_dim
        self.Hard = hard
        self.Space = Modules.LinearEncoder(input_dim, hidden_dim, latent_dim, dropout)
        self.Time = Modules.LinearEncoder(sequence_length, hidden_dim, 1, dropout)

    def forward(self, z, noise=None):
        logits = self.Time(self.Space(z).swapaxes(1, 2)).squeeze(-1)
        p = Manifolds.softmax(logits, self.CodeDim)
        z = Manifolds.gumbel(
            logits, self.CodeDim, noise=noise, temperature=1.0, hard=self.Hard
        )
        return z, p


class Estimator(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim, code_dim, hard, dropout):
        super(Estimator, self).__init__()
        self.CodeDim = code_dim
        self.Hard = hard
        self.Layers = Modules.LinearEncoder(input_dim, hidden_dim, latent_dim, dropout)

    def forward(self, x, noise=None):
        logits = self.Layers(x)
        p = Manifolds.softmax(logits, self.CodeDim)
        z = Manifolds.gumbel(
            logits, self.CodeDim, noise=noise, temperature=1.0, hard=self.Hard
        )
        return z, p


class Denoiser(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim, code_dim, hard, dropout):
        super(Denoiser, self).__init__()
        self.CodeDim = code_dim
        self.Hard = hard
        self.Layers = Modules.LinearEncoder(input_dim, hidden_dim, latent_dim, dropout)

    def forward(self, seed, x, noise=None):
        x = torch.cat((seed, x), -1)
        logits = self.Layers(x)
        p = Manifolds.softmax(logits, self.CodeDim)
        z = Manifolds.gumbel(
            logits, self.CodeDim, noise=noise, temperature=1.0, hard=self.Hard
        )
        return z, p


class Decoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout):
        super(Decoder, self).__init__()
        self.Layers = Modules.LinearFiLMEncoder(
            input_dim, hidden_dim, output_dim, 1, dropout
        )

    def forward(self, code, x, timestamps):
        z = torch.cat((code, x), dim=-1)
        z = z.unsqueeze(1).repeat(1, timestamps.shape[0], 1)
        t = timestamps.reshape(1, -1, 1)
        return self.Layers(z, t)


# Inputs=[Batch, InputDim]
# Outputs=[Batch, Sequence, OutputDim]
class Model(nn.Module):
    def __init__(
        self,
        input_dim,
        output_dim,
        regularization_dim,
        sequence_length,
        sequence_window,
        encoder_dim,
        estimator_dim,
        codebook_channels,
        codebook_dims,
        decoder_dim,
        dropout,
        hard,
        plotting,
    ):
        super(Model, self).__init__()

        self.InputDim = input_dim
        self.OutputDim = output_dim
        self.RegularizationDim = regularization_dim
        self.SequenceLength = sequence_length
        self.SequenceWindow = sequence_window
        self.LatentDim = codebook_channels * codebook_dims

        self.InputStats = Stats.RunningStats(input_dim)
        self.OutputStats = Stats.RunningStats(output_dim)
        self.RegularizationStats = Stats.RunningStats(regularization_dim)
        self.TimeStats = Stats.RunningStats(1)
        for _ in range(100):
            self.TimeStats.Update(self.timing())

        self.Encoder = Encoder(
            output_dim + regularization_dim,
            encoder_dim,
            self.LatentDim,
            codebook_dims,
            sequence_length,
            hard,
            dropout,
        )
        self.Estimator = Estimator(
            input_dim, estimator_dim, self.LatentDim, codebook_dims, hard, dropout
        )
        self.Denoiser = Denoiser(
            self.LatentDim + input_dim,
            estimator_dim,
            self.LatentDim,
            codebook_dims,
            hard,
            dropout,
        )
        self.Decoder = Decoder(
            self.LatentDim + input_dim, decoder_dim, output_dim, dropout
        )
        self.Regularizer = Decoder(
            self.LatentDim + input_dim, decoder_dim, regularization_dim, dropout
        )

        self.Plotting = plotting
        if self.Plotting > 0:
            self.Step = 0
            plt.ion()
            _, self.AxManifold = plt.subplots(1, 3, figsize=(12, 4))

    def forward(
        self, input, timestamps=None, noise=None, iterations=0, seed=None, results=None
    ):
        input = self.InputStats.Normalize(input)
        timestamps = self.timing() if timestamps is None else timestamps
        timestamps = timestamps.to(input.device)
        timestamps = self.TimeStats.Normalize(timestamps)

        z, p = self.Estimator(input, noise=noise)

        if results is not None:
            results.append(self.evaluate(z, input, timestamps))
        if iterations > 0:
            seed = seed if seed is not None else torch.randn_like(p)
            p += seed
            for _ in range(iterations):
                z, p = self.Denoiser(p, input, noise=noise)
                if results is not None:
                    results.append(self.evaluate(z, input, timestamps))

        y = self.Decoder(z, input, timestamps)
        y = self.OutputStats.Denormalize(y)

        return y, z, p, results

    def evaluate(self, z, input, timestamps):
        return self.OutputStats.Denormalize(self.Decoder(z, input, timestamps))

    def reconstruct(self, input, output, regularization):
        input = self.InputStats.Normalize(input)
        output = self.OutputStats.Normalize(output)
        regularization = self.RegularizationStats.Normalize(regularization)
        timestamps = self.TimeStats.Normalize(self.timing().to(output.device))

        target, _ = self.Encoder(torch.cat((output, regularization), -1))
        pred = self.Decoder(target, input, timestamps)

        return self.OutputStats.Denormalize(pred)

    def learn(self, input, output, regularization, update_stats):
        input = (
            self.InputStats.UpdateAndNormalize(input)
            if update_stats
            else self.InputStats.Normalize(input)
        )
        output = (
            self.OutputStats.UpdateAndNormalize(output)
            if update_stats
            else self.OutputStats.Normalize(output)
        )
        regularization = (
            self.RegularizationStats.UpdateAndNormalize(regularization)
            if update_stats
            else self.RegularizationStats.Normalize(regularization)
        )
        timestamps = self.TimeStats.Normalize(self.timing().to(output.device))

        self.eval()
        with torch.no_grad():
            _, destination = self.Encoder(torch.cat((output, regularization), -1))
            _, source = self.Estimator(input)
        self.train()

        code, _ = self.Encoder(torch.cat((output, regularization), -1))
        pred = self.Decoder(code, input, timestamps)
        reg = self.Regularizer(code, input, timestamps)
        _, estimate = self.Estimator(input)

        U = torch.rand(1).to(destination.device)
        seed = (1 - U) * destination + U * (
            source.detach() + torch.randn_like(destination)
        )
        _, denoised = self.Denoiser(seed, input)

        mse = nn.MSELoss()
        result = {
            "Y": self.OutputStats.Denormalize(pred),
            "R": self.RegularizationStats.Denormalize(reg),
        }
        loss = {
            "Reconstruction Loss": mse(pred, output),
            "Regularization Loss": mse(reg, regularization),
            "Matching Loss": mse(estimate, destination),
            "Denoising Loss": mse(denoised, destination),
        }

        # Visualize
        if self.Plotting > 0:
            self.Step += 1
            if self.Step == self.Plotting:
                self.Step = 0
                self.eval()
                with torch.no_grad():
                    _, target = self.Encoder(torch.cat((output, regularization), -1))
                    _, estimate = self.Estimator(input)
                    denoised = estimate + torch.randn_like(estimate)
                    iterations = 10
                    for _ in range(iterations):
                        _, denoised = self.Denoiser(denoised, input)
                    Plotting.PCA2D(
                        ax=self.AxManifold[0], values=target, title="Target Manifold"
                    )
                    Plotting.PCA2D(
                        ax=self.AxManifold[1],
                        values=estimate,
                        title="Estimate Manifold",
                    )
                    Plotting.PCA2D(
                        ax=self.AxManifold[2],
                        values=denoised,
                        title="Denoised Manifold",
                    )
                self.train()
                plt.tight_layout()
                plt.show()
                plt.gcf().canvas.draw()
                plt.gcf().canvas.start_event_loop(1e-1)

        return result, loss

    def timing(self):
        return torch.linspace(0.0, self.SequenceWindow, self.SequenceLength)


if __name__ == "__main__":
    from ai4animation import generators, plotting, utility

    # Toy Example Code
    features = 10
    sequence_length = 1
    iterations = 10
    noise = torch.rand(128)
    # seed = torch.randn(128)
    seed = None
    SAMPLE_COUNT = 10000
    BATCH_SIZE = 32
    DRAW_INTERVAL = 100

    model = Model(
        input_dim=1,
        output_dim=int(features / sequence_length),
        sequence_length=sequence_length,
        sequence_window=0.5,
        encoder_dim=64,
        estimator_dim=64,
        codebook_channels=16,
        codebook_dims=8,
        decoder_dim=64,
        dropout=0.1,
        hard=False,
        plotting=DRAW_INTERVAL,
    )
    optimizer_prior = torch.optim.Adam(
        list(model.Encoder.parameters()) + list(model.Decoder.parameters()), 1e-3
    )
    optimizer_matcher = torch.optim.Adam(list(model.Estimator.parameters()), 1e-3)
    optimizer_denoiser = torch.optim.Adam(list(model.Denoiser.parameters()), 1e-3)

    loss_history = utility.PlottingWindow(
        "Loss History", drawInterval=DRAW_INTERVAL, yScale="log"
    )

    def generate_data(size, min, max):
        X1, Y1 = generators.AmbiguousSquareFunctions(size, features, min, max)
        X2, Y2 = generators.AmbiguousSineFunctions(size, features, min, max)
        X = torch.cat((X1, X2), dim=0)
        Y = torch.cat((Y1, Y2), dim=0)
        Y = Y.reshape(-1, sequence_length, int(features / sequence_length))
        return (X, Y)

    plots = None
    for i in range(0, SAMPLE_COUNT):
        print("Progress", round(100 * i / SAMPLE_COUNT, 2), "%", end="\r")

        X, Y = generate_data(BATCH_SIZE, 1.0, 4.0)

        loss = model.learn(X, Y, True)

        optimizer_prior.zero_grad()
        loss["Reconstruction Loss"].backward()
        optimizer_prior.step()

        optimizer_matcher.zero_grad()
        loss["Matching Loss"].backward()
        optimizer_matcher.step()

        optimizer_denoiser.zero_grad()
        loss["Denoising Loss"].backward()
        optimizer_denoiser.step()

        for k, v in loss.items():
            loss_history.Add((plotting.Item(v), k))

        # PLOTTING
        plt.ion()
        plots = (
            plots
            if plots is not None
            else plt.subplots(4, iterations + 3, figsize=(20, 8))
        )
        fig, axes = plots

        def fn(scale, axis):
            xTrue = scale * torch.ones_like(X) if scale is not None else X
            yTrue = (
                generate_data(BATCH_SIZE, scale, scale)[1] if scale is not None else Y
            )
            _, _, _, results = model(
                xTrue,
                timestamps=None,
                noise=noise,
                iterations=iterations,
                seed=seed,
                results=[],
            )
            results.insert(0, yTrue)
            results.insert(1, model.reconstruct(xTrue, yTrue))
            for k in range(len(results)):
                plotting.PlotFunctions(
                    axes[axis, k],
                    results[k].detach().flatten(start_dim=1),
                    "Iter=" + str(k - 2) + " X=" + str(scale) + " "
                    if k > 1
                    else "Y True"
                    if k == 0
                    else "Y Rec",
                    step=1,
                    yLimits=[-5, 5],
                )

        if i % DRAW_INTERVAL == 0:
            model.eval()
            with torch.no_grad():
                fn(None, 0)
                fn(1.0, 1)
                fn(2.0, 2)
                fn(4.0, 3)
            model.train()
            plt.tight_layout()
            plt.show()
            plt.gcf().canvas.draw()
            plt.gcf().canvas.start_event_loop(1e-1)
