# Copyright (c) Meta Platforms, Inc. and affiliates.
import torch
import torch.nn as nn
from ai4animation.AI import Manifolds, Modules, Stats

TIME_SAMPLES = 100


class Encoder(nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_dim,
        latent_dim,
        sequence_length,
        dropout,
    ):
        super(Encoder, self).__init__()
        self.Space = Modules.LinearEncoder(input_dim, hidden_dim, latent_dim, dropout)
        self.Time = Modules.LinearEncoder(sequence_length, hidden_dim, 1, dropout)

    def forward(self, z):
        return self.Time(self.Space(z).swapaxes(1, 2)).squeeze(-1)


class Decoder(nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_dim,
        output_dim,
        regularization_dim,
        dropout,
    ):
        super(Decoder, self).__init__()
        if regularization_dim is not None:
            self.Layers = Modules.RegularizedLinearFiLMEncoder(
                input_dim,
                hidden_dim,
                output_dim,
                regularization_dim,
                1,
                dropout,
            )
        else:
            self.Layers = Modules.LinearFiLMEncoder(
                input_dim,
                hidden_dim,
                output_dim,
                1,
                dropout,
            )

    def forward(self, codes, timestamps):
        z = codes
        z = z.unsqueeze(1).repeat(1, timestamps.shape[0], 1)
        film = timestamps.reshape(1, -1, 1).repeat(codes.shape[0], 1, 1)
        return self.Layers(z, film)


class Estimator(nn.Module):
    def __init__(
        self, input_dim, hidden_dim, latent_dim, channels, dimensions, dropout
    ):
        super(Estimator, self).__init__()
        # self.Layers = Modules.LinearEncoder(input_dim, hidden_dim, latent_dim, dropout)
        self.Layers = Modules.CategoricalEncoder(
            input_dim, hidden_dim, latent_dim, channels, dimensions, dropout
        )

        # self.C = channels
        # self.D = dimensions

        # self.Encoder = Modules.LinearEncoder(
        #     input_dim, hidden_dim, channels * dimensions, dropout
        # )
        # self.Decoder = Modules.LinearEncoder(
        #     channels * dimensions, hidden_dim, latent_dim, dropout
        # )

    def forward(self, x):
        return self.Layers(x)

    # def forward(self, z):
    #     z = self.Encoder(z)
    #     if self.training:
    #         z = Manifolds.gumbel(z, self.D)
    #     else:
    #         z = Manifolds.softmax(z, self.D)
    #     z = self.Decoder(z)
    #     return z


class Denoiser(nn.Module):
    def __init__(self, input_dim, latent_dim, dropout):
        super(Denoiser, self).__init__()
        self.Layers = Modules.LinearEncoder(input_dim, latent_dim, latent_dim, dropout)

    def forward(self, seed, x):
        return self.Layers(torch.cat((seed, x), -1))


# Outputs=[Batch, Sequence, OutputDim]
# Regularizations=[Batch, Sequence, RegularizationDim]
class Prior(nn.Module):
    def __init__(
        self,
        output_dim,
        regularization_dim,
        sequence_length,
        sequence_window,
        encoder_dim,
        codebook_channels,
        codebook_dims,
        decoder_dim,
        dropout,
    ):
        super(Prior, self).__init__()

        self.OutputDim = output_dim
        self.RegularizationDim = regularization_dim
        self.SequenceLength = sequence_length
        self.SequenceWindow = sequence_window
        self.LatentDim = codebook_channels * codebook_dims
        self.Channels = codebook_channels
        self.Dims = codebook_dims

        self.OutputStats = Stats.RunningStats(output_dim)
        self.RegularizationStats = Stats.RunningStats(regularization_dim)
        self.TimeStats = Stats.RunningStats(1)
        for _ in range(TIME_SAMPLES):
            self.TimeStats.Update(self.timing())

        self.Encoder = Encoder(
            output_dim + regularization_dim,
            encoder_dim,
            self.LatentDim,
            sequence_length,
            dropout,
        )
        self.Decoder = Decoder(
            self.LatentDim,
            decoder_dim,
            output_dim,
            regularization_dim,
            dropout,
        )

    def encode(self, output, regularization):
        output = self.OutputStats.Normalize(output)
        regularization = self.RegularizationStats.Normalize(regularization)
        logits = self.Encoder(torch.cat((output, regularization), -1))
        return logits

    def decode(self, codes, timestamps=None):
        timestamps = self.TimeStats.Normalize(
            (self.timing().to(codes.device) if timestamps is None else timestamps)
        )
        if self.training:
            pred, reg = self.Decoder(codes, timestamps)
            return self.OutputStats.Denormalize(
                pred
            ), self.RegularizationStats.Denormalize(reg)
        else:
            pred = self.Decoder(codes, timestamps)
            return self.OutputStats.Denormalize(pred)

    def manifold(self, logits, sample):
        if sample:
            return Manifolds.gumbel(logits, self.Dims)
        else:
            return Manifolds.softmax(logits, self.Dims)

    def reconstruct(self, output, regularization):
        z = self.encode(output, regularization)
        z = self.manifold(z, sample=False)
        z = self.decode(z, timestamps=None)
        return z

    def learn(self, output, regularization, update_stats):
        if update_stats:
            self.OutputStats.Update(output)
            self.RegularizationStats.Update(regularization)

        logits = self.encode(output, regularization)
        codes = self.manifold(logits, sample=True)
        pred, reg = self.decode(codes, timestamps=None)

        mse = nn.MSELoss()
        result = {
            "Y": pred,
            "R": reg,
        }
        loss = {
            "Reconstruction Loss": mse(
                self.OutputStats.Normalize(pred), self.OutputStats.Normalize(output)
            ),
            "Regularization Loss": mse(
                self.RegularizationStats.Normalize(reg),
                self.RegularizationStats.Normalize(regularization),
            ),
        }

        return result, loss

    def timing(self):
        return torch.linspace(0.0, self.SequenceWindow, self.SequenceLength)


# Inputs=[Batch, InputDim]
class Sampler(nn.Module):
    def __init__(self, prior, input_dim, estimator_dim, channels, dimensions, dropout):
        super(Sampler, self).__init__()

        self.Prior = prior

        self.InputDim = input_dim

        self.InputStats = Stats.RunningStats(input_dim)
        self.Estimator = Estimator(
            input_dim,
            estimator_dim,
            self.Prior.LatentDim,
            channels,
            dimensions,
            dropout,
        )
        self.Denoiser = Denoiser(
            self.Prior.LatentDim + input_dim,
            self.Prior.LatentDim,
            dropout,
        )

    def forward(self, input, timestamps=None, iterations=0):
        input = self.InputStats.Normalize(input)

        z = self.Prior.manifold(self.Estimator(input), False)  # Softmax Probabilities

        if iterations > 0:
            for _ in range(iterations):
                z = self.Prior.manifold(
                    self.Denoiser(z, input), False
                )  # Softmax Probabilities

        y = self.Prior.decode(z, timestamps)

        return y

    def learn(self, input, output, regularization, update_stats):
        if update_stats:
            self.InputStats.Update(input)
        input = self.InputStats.Normalize(input)

        self.eval()
        with torch.no_grad():
            target = self.Prior.manifold(
                self.Prior.encode(output, regularization), False
            ).detach()  # SoftMax Probabilities
            source = self.Prior.manifold(
                self.Estimator(input), True
            ).detach()  # Gumbel Probabilities
        self.train()

        # Denoiser
        U = torch.rand(1, device=target.device)
        seed = (1 - U) * target + U * source
        denoised = self.Prior.manifold(
            self.Denoiser(seed, input), False
        )  # Softmax Probabilities

        # Estimator
        estimate = self.Prior.manifold(
            self.Estimator(input), False
        )  # Softmax Probabilities

        mse = nn.MSELoss()
        result = None
        loss = {
            "Matching Loss": mse(estimate, target),
            "Denoising Loss": mse(denoised, target),
        }

        return result, loss


# Inputs=[Batch, InputDim]
# Outputs=[Batch, Sequence, OutputDim]
# Regularizations=[Batch, Sequence, RegularizationDim]
class Model(nn.Module):
    def __init__(
        self,
        input_dim,
        output_dim,
        regularization_dim,
        sequence_length,
        sequence_window,
        encoder_dim,
        codebook_channels,
        codebook_dims,
        decoder_dim,
        estimator_dim,
        dropout,
    ):
        super(Model, self).__init__()

        self.InputDim = input_dim
        self.OutputDim = output_dim
        self.SequenceLength = sequence_length
        self.SequenceWindow = sequence_window
        self.LatentDim = codebook_channels * codebook_dims
        self.Channels = codebook_channels
        self.Dims = codebook_dims

        self.InputStats = Stats.RunningStats(input_dim)
        self.OutputStats = Stats.RunningStats(output_dim)
        self.RegularizationStats = (
            None
            if regularization_dim is None
            else Stats.RunningStats(regularization_dim)
        )
        self.TimeStats = Stats.RunningStats(1)
        for _ in range(TIME_SAMPLES):
            self.TimeStats.Update(self.timing())

        self.Encoder = Encoder(
            output_dim + (0 if regularization_dim is None else regularization_dim),
            encoder_dim,
            self.LatentDim,
            sequence_length,
            dropout,
        )
        self.Decoder = Decoder(
            self.LatentDim,
            decoder_dim,
            output_dim,
            regularization_dim,
            dropout,
        )
        self.Estimator = Estimator(
            input_dim,
            estimator_dim,
            self.LatentDim,
            codebook_channels,
            codebook_dims,
            dropout,
        )
        self.Denoiser = Denoiser(
            self.LatentDim + input_dim,
            self.LatentDim,
            dropout,
        )

    def forward(self, x, iterations=1, sample=False):
        x = self.InputStats.Normalize(x)

        z = self.Estimator(x)
        c = self.manifold(z, sample=sample)

        for _ in range(iterations):
            c = self.manifold(self.Denoiser(c, x), False)  # Softmax Probabilities

        y, _ = self.decode(c, timestamps=None)
        return y

    def learn(self, input, output, update_stats, regularization=None):
        if update_stats:
            self.InputStats.Update(input)
            self.OutputStats.Update(output)
            if self.RegularizationStats is not None:
                self.RegularizationStats.Update(regularization)

        # Autoencoder
        logits = self.encode(output, regularization)
        codes = self.manifold(logits, sample=True)
        pred, reg = self.decode(codes, timestamps=None)

        # Target
        target = self.manifold(logits, sample=False)  # Softmax Probabilities

        # Normalize
        input = self.InputStats.Normalize(input)

        # Estimator
        estimate = self.manifold(self.Estimator(input), False)  # Softmax Probabilities

        # Denoiser
        U = torch.rand(1, device=target.device)
        seed = (1 - U) * target + U * codes
        denoised = self.manifold(
            self.Denoiser(seed, input), False
        )  # Softmax Probabilities

        # Losses
        loss = {}
        mse = nn.MSELoss()
        loss["Reconstruction Loss"] = mse(
            self.OutputStats.Normalize(pred), self.OutputStats.Normalize(output)
        )
        if reg is not None:
            loss["Regularization Loss"] = mse(
                self.RegularizationStats.Normalize(reg),
                self.RegularizationStats.Normalize(regularization),
            )
        loss["Matching Loss"] = mse(estimate, target)
        loss["Denoising Loss"] = mse(denoised, target)
        return loss

    def encode(self, output, regularization):
        output = self.OutputStats.Normalize(output)
        if regularization is not None:
            regularization = self.RegularizationStats.Normalize(regularization)
            logits = self.Encoder(torch.cat((output, regularization), -1))
        else:
            logits = self.Encoder(output)
        return logits

    def decode(self, codes, timestamps=None):
        timestamps = self.TimeStats.Normalize(
            (self.timing().to(codes.device) if timestamps is None else timestamps)
        )
        if self.training and self.RegularizationStats is not None:
            pred, reg = self.Decoder(codes, timestamps)
            return self.OutputStats.Denormalize(
                pred
            ), self.RegularizationStats.Denormalize(reg)
        else:
            pred = self.Decoder(codes, timestamps)
            return self.OutputStats.Denormalize(pred), None

    def reconstruct(self, output, regularization=None):
        z = self.encode(output, regularization)
        z = self.manifold(z, sample=False)
        z, _ = self.decode(z, timestamps=None)
        return z

    def manifold(self, logits, sample):
        if sample:
            return Manifolds.gumbel(logits, self.Dims)
        else:
            return Manifolds.softmax(logits, self.Dims)

    def timing(self):
        return torch.linspace(0.0, self.SequenceWindow, self.SequenceLength)
