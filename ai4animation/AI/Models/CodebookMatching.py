# Copyright (c) Meta Platforms, Inc. and affiliates.
import torch
import torch.nn as nn
from ai4animation.AI.Library import Losses
from ai4animation.AI.Library.Blocks import LinearBlock, RegularizedFiLMLinearBlock
from ai4animation.AI.Library.Layers import CodebookLayer
from ai4animation.AI.Library.Statistics import RunningStatistics
from ai4animation.AI.Models import CategoricalEncoderDecoder, MultiLayerPerceptron

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
        self.Space = LinearBlock(input_dim, hidden_dim, latent_dim, dropout)
        self.Time = LinearBlock(sequence_length, hidden_dim, 1, dropout)

    def forward(self, z):
        return self.Time(self.Space(z).swapaxes(1, 2)).squeeze(-1)


class Decoder(nn.Module):
    def __init__(
        self,
        latent_dim,
        hidden_dim,
        output_dim,
        regularization_dim,
        dropout,
    ):
        super(Decoder, self).__init__()
        self.Layers = RegularizedFiLMLinearBlock(
            latent_dim,
            hidden_dim,
            output_dim,
            regularization_dim,
            1,
            dropout,
        )

    def forward(self, codes, timestamps):
        z = codes
        z = z.unsqueeze(1).repeat(1, timestamps.shape[0], 1)
        film = timestamps.reshape(1, -1, 1).repeat(codes.shape[0], 1, 1)
        return self.Layers(z, film)


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
        codebook_classes,
        decoder_dim,
        estimator_dim,
        denoiser_dim,
        dropout,
    ):
        super(Model, self).__init__()

        self.SequenceLength = sequence_length
        self.SequenceWindow = sequence_window

        self.InputStatistics = RunningStatistics(input_dim)
        self.OutputStatistics = RunningStatistics(output_dim)
        self.RegularizationStatistics = RunningStatistics(regularization_dim)
        self.TimeStatistics = RunningStatistics(1)
        for _ in range(TIME_SAMPLES):
            self.TimeStatistics.Update(self.timing())

        self.Codebook = CodebookLayer(codebook_channels, codebook_classes)
        self.Encoder = Encoder(
            output_dim + regularization_dim,
            encoder_dim,
            self.Codebook.dimensions(),
            sequence_length,
            dropout,
        )
        self.Decoder = Decoder(
            self.Codebook.dimensions(),
            decoder_dim,
            output_dim,
            regularization_dim,
            dropout,
        )
        self.Estimator = CategoricalEncoderDecoder.Model(
            input_dim,
            self.Codebook.dimensions(),
            estimator_dim,
            codebook_channels,
            codebook_classes,
            estimator_dim,
            dropout,
        )
        self.Denoiser = MultiLayerPerceptron.Model(
            self.Codebook.dimensions() + input_dim,
            self.Codebook.dimensions(),
            denoiser_dim,
            dropout,
        )

    def input_dim(self):
        return self.Estimator.input_dim()

    def output_dim(self):
        return self.Decoder.output_dim()

    def forward(self, x, iterations=1):
        t = self.TimeStatistics.Normalize(self.timing().to(x.device))
        x = self.InputStatistics.Normalize(x)
        z = self.Estimator(x)
        c = self.Codebook(z, sample=False)
        for _ in range(iterations):
            c = self.Codebook(self.Denoiser(torch.cat((c, x), -1)), sample=False)
        y = self.Decoder(c, t)
        y = self.OutputStatistics.Denormalize(y)
        return y

    def learn(self, input, output, regularization, update_statistics):
        if update_statistics:
            self.InputStatistics.Update(input)
            self.OutputStatistics.Update(output)
            self.RegularizationStatistics.Update(regularization)

        # Normalize
        input = self.InputStatistics.Normalize(input)
        output = self.OutputStatistics.Normalize(output)
        regularization = self.RegularizationStatistics.Normalize(regularization)
        timestamps = self.TimeStatistics.Normalize(self.timing().to(input.device))

        # Prior
        logits = self.Encoder(torch.cat((output, regularization), -1))
        codes = self.Codebook(logits, sample=True)
        pred, reg = self.Decoder(codes, timestamps)

        # Target
        target = self.Codebook(logits, sample=False)

        # Estimate
        estimate = self.Codebook(self.Estimator(input), sample=False)

        # Denoise
        U = torch.rand(1, device=target.device)
        seed = (1 - U) * target + U * codes
        denoised = self.Codebook(
            self.Denoiser(torch.cat((seed, input), -1)), sample=False
        )

        # Losses
        loss = {}
        loss["Reconstruction Loss"] = Losses.MSE(pred, output)
        loss["Regularization Loss"] = Losses.MSE(reg, regularization)
        loss["Matching Loss"] = Losses.MSE(estimate, target)
        loss["Denoising Loss"] = Losses.MSE(denoised, target)

        return None, loss

    def timing(self):
        return torch.linspace(0.0, self.SequenceWindow, self.SequenceLength).reshape(
            -1, 1
        )


# # Outputs=[Batch, Sequence, OutputDim]
# # Regularizations=[Batch, Sequence, RegularizationDim]
# class Prior(nn.Module):
#     def __init__(
#         self,
#         output_dim,
#         regularization_dim,
#         sequence_length,
#         sequence_window,
#         encoder_dim,
#         codebook_channels,
#         codebook_dims,
#         decoder_dim,
#         dropout,
#     ):
#         super(Prior, self).__init__()

#         self.OutputDim = output_dim
#         self.RegularizationDim = regularization_dim
#         self.SequenceLength = sequence_length
#         self.SequenceWindow = sequence_window
#         self.LatentDim = codebook_channels * codebook_dims
#         self.Channels = codebook_channels
#         self.Dims = codebook_dims

#         self.OutputStats = Stats.RunningStats(output_dim)
#         self.RegularizationStats = Stats.RunningStats(regularization_dim)
#         self.TimeStats = Stats.RunningStats(1)
#         for _ in range(TIME_SAMPLES):
#             self.TimeStats.Update(self.timing())

#         self.Encoder = Encoder(
#             output_dim + regularization_dim,
#             encoder_dim,
#             self.LatentDim,
#             sequence_length,
#             dropout,
#         )
#         self.Decoder = Decoder(
#             self.LatentDim,
#             decoder_dim,
#             output_dim,
#             regularization_dim,
#             dropout,
#         )

#     def encode(self, output, regularization):
#         output = self.OutputStats.Normalize(output)
#         regularization = self.RegularizationStats.Normalize(regularization)
#         logits = self.Encoder(torch.cat((output, regularization), -1))
#         return logits

#     def decode(self, codes, timestamps=None):
#         timestamps = self.TimeStats.Normalize(
#             (self.timing().to(codes.device) if timestamps is None else timestamps)
#         )
#         if self.training:
#             pred, reg = self.Decoder(codes, timestamps)
#             return self.OutputStats.Denormalize(
#                 pred
#             ), self.RegularizationStats.Denormalize(reg)
#         else:
#             pred = self.Decoder(codes, timestamps)
#             return self.OutputStats.Denormalize(pred)

#     def manifold(self, logits, sample):
#         if sample:
#             return Manifolds.gumbel(logits, self.Dims)
#         else:
#             return Manifolds.softmax(logits, self.Dims)

#     def reconstruct(self, output, regularization):
#         z = self.encode(output, regularization)
#         z = self.manifold(z, sample=False)
#         z = self.decode(z, timestamps=None)
#         return z

#     def learn(self, output, regularization, update_stats):
#         if update_stats:
#             self.OutputStats.Update(output)
#             self.RegularizationStats.Update(regularization)

#         logits = self.encode(output, regularization)
#         codes = self.manifold(logits, sample=True)
#         pred, reg = self.decode(codes, timestamps=None)

#         mse = nn.MSELoss()
#         result = {
#             "Y": pred,
#             "R": reg,
#         }
#         loss = {
#             "Reconstruction Loss": mse(
#                 self.OutputStats.Normalize(pred), self.OutputStats.Normalize(output)
#             ),
#             "Regularization Loss": mse(
#                 self.RegularizationStats.Normalize(reg),
#                 self.RegularizationStats.Normalize(regularization),
#             ),
#         }

#         return result, loss

#     def timing(self):
#         return torch.linspace(0.0, self.SequenceWindow, self.SequenceLength)


# # Inputs=[Batch, InputDim]
# class Sampler(nn.Module):
#     def __init__(self, prior, input_dim, estimator_dim, channels, dimensions, dropout):
#         super(Sampler, self).__init__()

#         self.Prior = prior

#         self.InputDim = input_dim

#         self.InputStats = Stats.RunningStats(input_dim)
#         self.Estimator = Estimator(
#             input_dim,
#             estimator_dim,
#             self.Prior.LatentDim,
#             channels,
#             dimensions,
#             dropout,
#         )
#         self.Denoiser = Denoiser(
#             self.Prior.LatentDim + input_dim,
#             self.Prior.LatentDim,
#             dropout,
#         )

#     def forward(self, input, timestamps=None, iterations=0):
#         input = self.InputStats.Normalize(input)

#         z = self.Prior.manifold(self.Estimator(input), False)  # Softmax Probabilities

#         if iterations > 0:
#             for _ in range(iterations):
#                 z = self.Prior.manifold(
#                     self.Denoiser(z, input), False
#                 )  # Softmax Probabilities

#         y = self.Prior.decode(z, timestamps)

#         return y

#     def learn(self, input, output, regularization, update_stats):
#         if update_stats:
#             self.InputStats.Update(input)
#         input = self.InputStats.Normalize(input)

#         self.eval()
#         with torch.no_grad():
#             target = self.Prior.manifold(
#                 self.Prior.encode(output, regularization), False
#             ).detach()  # SoftMax Probabilities
#             source = self.Prior.manifold(
#                 self.Estimator(input), True
#             ).detach()  # Gumbel Probabilities
#         self.train()

#         # Denoiser
#         U = torch.rand(1, device=target.device)
#         seed = (1 - U) * target + U * source
#         denoised = self.Prior.manifold(
#             self.Denoiser(seed, input), False
#         )  # Softmax Probabilities

#         # Estimator
#         estimate = self.Prior.manifold(
#             self.Estimator(input), False
#         )  # Softmax Probabilities

#         mse = nn.MSELoss()
#         result = None
#         loss = {
#             "Matching Loss": mse(estimate, target),
#             "Denoising Loss": mse(denoised, target),
#         }

#         return result, loss
