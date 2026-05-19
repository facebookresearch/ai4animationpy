# Copyright (c) Meta Platforms, Inc. and affiliates.
import torch
import torch.nn as nn
from ai4animation.AI.Library import Losses
from ai4animation.AI.Library.Blocks import FiLMLinearBlock, LinearBlock, SpaceTimeBlock
from ai4animation.AI.Library.Layers import CodebookLayer
from ai4animation.AI.Library.Statistics import RunningStatistics
from ai4animation.AI.Models import CategoricalEncoderDecoder, MultiLayerPerceptron


# Codebook Matching Model
class Model(nn.Module):
    def __init__(
        self,
        prior,
        sampler,
    ):
        super(Model, self).__init__()
        self.Prior = prior
        self.Sampler = sampler

    def input_dim(self):
        return self.Sampler.input_dim()

    def forward(self, x, iterations=1, sample=False):
        return self.Prior(self.Sampler(x, iterations, sample))

    def learn(self, input, output, update_statistics):
        prior_result, prior_loss = self.Prior.learn(output, update_statistics)
        sampler_result, sampler_loss = self.Sampler.learn(
            input, prior_result["Z"], update_statistics
        )
        result = prior_result | sampler_result
        loss = prior_loss | sampler_loss
        return result, loss


# Codebook Matching Prior
class Prior(nn.Module):
    def __init__(self, encoder, decoder, codebook):
        super(Prior, self).__init__()
        self.FeatureStatistics = RunningStatistics(decoder.output_dim())
        self.Encoder = encoder
        self.Decoder = decoder
        self.Codebook = codebook

    def forward(self, codes):
        y = self.decode(codes)
        y = self.FeatureStatistics.Denormalize(y)
        return y

    def target(self, y):
        y = self.FeatureStatistics.Normalize(y)
        z = self.encode(y)
        return z

    def encode(self, x):
        return self.Encoder(x)

    def decode(self, x):
        return self.Decoder(x)

    def reconstruct(self, y, sample):
        y = self.FeatureStatistics.Normalize(y)
        z = self.encode(y)
        c = self.Codebook(z, sample=sample)
        y = self.decode(c)
        y = self.FeatureStatistics.Denormalize(y)
        return y

    def learn(self, features, update_statistics):
        if update_statistics:
            self.FeatureStatistics.Update(features)
        features = self.FeatureStatistics.Normalize(features)
        logits = self.encode(features)
        codes = self.Codebook(logits, sample=True)
        pred = self.decode(codes)
        result = {
            "Z": logits,
            "C": codes,
            "Y": self.FeatureStatistics.Denormalize(pred),
        }
        loss = {
            "Reconstruction Loss": Losses.MSE(pred, features),
        }
        return result, loss


# Codebook Matching Sampler
class Sampler(nn.Module):
    def __init__(self, estimator, denoiser, codebook):
        super(Sampler, self).__init__()
        self.FeatureStatistics = RunningStatistics(estimator.input_dim())
        self.Estimator = estimator
        self.Denoiser = denoiser
        self.Codebook = codebook

    def input_dim(self):
        return self.Estimator.input_dim()

    def forward(self, x, iterations=1, sample=False):
        x = self.FeatureStatistics.Normalize(x)
        z = self.estimate(x)
        c = self.Codebook(z, sample=sample)
        for _ in range(iterations):
            c = self.Codebook(self.denoise(c, x), sample=False)
        return c

    def estimate(self, x):
        return self.Estimator(x)

    def denoise(self, z, x):
        return self.Denoiser(torch.cat((z, x), -1))

    def learn(self, features, targets, update_statistics):
        if update_statistics:
            self.FeatureStatistics.Update(features)
        features = self.FeatureStatistics.Normalize(features)

        target_codes = self.Codebook(targets, sample=False)
        noised_codes = self.Codebook(targets, sample=True)

        # Estimate
        estimate_codes = self.Codebook(self.estimate(features), sample=False)

        # Denoise
        U = torch.rand(1, device=features.device)
        seed = (1 - U) * target_codes + U * noised_codes
        denoised_codes = self.Codebook(self.denoise(seed, features), sample=False)

        result = {}
        loss = {
            "Matching Loss": Losses.MSE(estimate_codes, target_codes),
            "Denoising Loss": Losses.MSE(denoised_codes, target_codes),
        }

        return result, loss


####################################################################################################
####################################################################################################
####################################################################################################
# Priors
####################################################################################################
####################################################################################################
####################################################################################################
class VanillaPrior(Prior):
    def __init__(
        self,
        feature_dim,
        encoder_dim,
        decoder_dim,
        codebook_channels,
        codebook_classes,
        dropout,
    ):
        super(VanillaPrior, self).__init__(
            encoder=LinearBlock(
                feature_dim,
                codebook_channels * codebook_classes,
                encoder_dim,
                dropout,
            ),
            decoder=LinearBlock(
                codebook_channels * codebook_classes,
                feature_dim,
                decoder_dim,
                dropout,
            ),
            codebook=CodebookLayer(codebook_channels, codebook_classes),
        )


class MotionPrior(Prior):
    def __init__(
        self,
        sequence_length,
        feature_dim,
        encoder_dim,
        decoder_dim,
        codebook_channels,
        codebook_classes,
        dropout,
    ):
        super(MotionPrior, self).__init__(
            encoder=SpaceTimeBlock(
                feature_dim,
                codebook_channels * codebook_classes,
                encoder_dim,
                sequence_length,
                1,
                dropout,
            ),
            decoder=FiLMLinearBlock(
                codebook_channels * codebook_classes,
                feature_dim,
                decoder_dim,
                1,
                dropout,
            ),
            codebook=CodebookLayer(codebook_channels, codebook_classes),
        )

        self.SequenceLength = sequence_length

        self.TimeStatistics = RunningStatistics(1)
        for _ in range(100):
            self.TimeStatistics.Update(self.timing())

    def timing(self, device=None):
        if device is not None:
            t = torch.linspace(0.0, 1.0, self.SequenceLength, device=device)
        else:
            t = torch.linspace(0.0, 1.0, self.SequenceLength)
        return t.reshape(-1, 1)

    def encode(self, x):
        return self.Encoder(x).squeeze(-2)

    def decode(self, x):
        x = x.unsqueeze(1).repeat(1, self.SequenceLength, 1)
        t = self.TimeStatistics.Normalize(self.timing(x.device))
        t = t.reshape(1, -1, 1).repeat(x.shape[0], 1, 1)
        return self.Decoder(x, t)


####################################################################################################
####################################################################################################
####################################################################################################
# Samplers
####################################################################################################
####################################################################################################
####################################################################################################
class VanillaSampler(Sampler):
    def __init__(
        self,
        feature_dim,
        estimator_dim,
        denoiser_dim,
        codebook_channels,
        codebook_classes,
        dropout,
    ):
        super(VanillaSampler, self).__init__(
            estimator=LinearBlock(
                feature_dim,
                codebook_channels * codebook_classes,
                estimator_dim,
                dropout,
            ),
            denoiser=LinearBlock(
                codebook_channels * codebook_classes + feature_dim,
                codebook_channels * codebook_classes,
                denoiser_dim,
                dropout,
            ),
            codebook=CodebookLayer(codebook_channels, codebook_classes),
        )


class MotionSampler(Sampler):
    def __init__(
        self,
        feature_dim,
        estimator_dim,
        denoiser_dim,
        codebook_channels,
        codebook_classes,
        dropout,
    ):
        super(MotionSampler, self).__init__(
            estimator=CategoricalEncoderDecoder.Model(
                feature_dim,
                codebook_channels * codebook_classes,
                estimator_dim,
                estimator_dim,
                codebook_channels,
                codebook_classes,
                dropout,
            ),
            denoiser=LinearBlock(
                codebook_channels * codebook_classes + feature_dim,
                codebook_channels * codebook_classes,
                denoiser_dim,
                dropout,
            ),
            codebook=CodebookLayer(codebook_channels, codebook_classes),
        )

####################################################################################################
####################################################################################################
####################################################################################################
# Models
####################################################################################################
####################################################################################################
####################################################################################################
# Input: [Batch, InputDim]
# Output: [Batch, OutputDim]
class VanillaModel(Model):
    def __init__(
        self,
        input_dim,
        output_dim,
        encoder_dim,
        decoder_dim,
        estimator_dim,
        denoiser_dim,
        codebook_channels,
        codebook_classes,
        dropout,
    ):
        super(VanillaModel, self).__init__(
            prior=VanillaPrior(
                output_dim,
                encoder_dim,
                decoder_dim,
                codebook_channels,
                codebook_classes,
                dropout,
            ),
            sampler=VanillaSampler(
                input_dim,
                estimator_dim,
                denoiser_dim,
                codebook_channels,
                codebook_classes,
                dropout,
            ),
        )


# Input: [Batch, InputDim]
# Output: [Batch, Sequence, OutputDim]
class MotionModel(Model):
    def __init__(
        self,
        sequence_length,
        input_dim,
        output_dim,
        encoder_dim,
        decoder_dim,
        estimator_dim,
        denoiser_dim,
        codebook_channels,
        codebook_classes,
        dropout,
    ):
        super(MotionModel, self).__init__(
            prior=MotionPrior(
                sequence_length,
                output_dim,
                encoder_dim,
                decoder_dim,
                codebook_channels,
                codebook_classes,
                dropout,
            ),
            sampler=MotionSampler(
                input_dim,
                estimator_dim,
                denoiser_dim,
                codebook_channels,
                codebook_classes,
                dropout,
            ),
        )

####################################################################################################
####################################################################################################
####################################################################################################
# Toy Example
####################################################################################################
####################################################################################################
####################################################################################################
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from ai4animation import Generators, Plotting

    # Toy Example Code
    FEATURES = 10
    SEQUENCE_LENGTH = 1
    EPOCH_COUNT = 150
    BATCH_COUNT = 1000
    BATCH_SIZE = 32
    ITERATIONS = 10
    TEST_SIZE = 100
    DRAW_INTERVAL = 100

    model = VanillaModel(
        input_dim=1,
        output_dim=FEATURES,
        encoder_dim=64,
        decoder_dim=64,
        estimator_dim=64,
        denoiser_dim=64,
        codebook_channels=16,
        codebook_classes=8,
        dropout=0.1,
    )

    optimizer = torch.optim.Adam(model.parameters())

    loss_history = Plotting.LossHistory(
        "Loss History", BATCH_COUNT, drawInterval=DRAW_INTERVAL, yScale="log"
    )

    def generate_data(batch, min, max):
        X1, Y1 = Generators.AmbiguousSquareFunctions(
            int(batch / 2), FEATURES * SEQUENCE_LENGTH, min, max
        )
        X2, Y2 = Generators.AmbiguousSineFunctions(
            int(batch / 2), FEATURES * SEQUENCE_LENGTH, min, max
        )
        X = torch.cat((X1, X2), dim=0)
        Y = torch.cat((Y1, Y2), dim=0)

        return (X, Y)

    plots = None
    for e in range(1, EPOCH_COUNT + 1, 1):
        print("Epoch", e)
        for i in range(0, BATCH_COUNT):
            print("Progress", round(100 * i / BATCH_COUNT, 2), "%", end="\r")

            X, Y = generate_data(BATCH_SIZE, 1.0, 4.0)

            _, loss = model.learn(X, Y, e == 1)

            optimizer.zero_grad()
            sum(loss.values()).backward()
            optimizer.step()

            loss_history.Add(loss)

            # PLOTTING
            plt.ion()
            plots = plots if plots is not None else plt.subplots(2, 3, figsize=(9, 6))
            fig, axes = plots
            if i % DRAW_INTERVAL == 0:
                model.eval()
                with torch.no_grad():
                    Xtrue, Ytrue = generate_data(TEST_SIZE, 1, 4)
                    Ypred = model(Xtrue, iterations=ITERATIONS, sample=True)
                    Yrec = model.Prior.reconstruct(Ytrue, sample=True)

                    Plotting.PlotFunctions(
                        axes[0, 0],
                        Ytrue.detach().flatten(start_dim=1),
                        "Ground Truth",
                        step=1,
                        yLimits=[-5, 5],
                    )

                    Plotting.PlotFunctions(
                        axes[0, 1],
                        Yrec.detach().flatten(start_dim=1),
                        "Reconstruction",
                        step=1,
                        yLimits=[-5, 5],
                    )

                    Plotting.PlotFunctions(
                        axes[0, 2],
                        Ypred.detach().flatten(start_dim=1),
                        "Sampling",
                        step=1,
                        yLimits=[-5, 5],
                    )

                    Plotting.PlotFunctions(
                        axes[1, 0],
                        model(
                            torch.zeros_like(Xtrue), iterations=ITERATIONS, sample=True
                        )
                        .detach()
                        .flatten(start_dim=1),
                        "X=0",
                        step=1,
                        yLimits=[-5, 5],
                    )

                    Plotting.PlotFunctions(
                        axes[1, 1],
                        model(
                            torch.ones_like(Xtrue), iterations=ITERATIONS, sample=True
                        )
                        .detach()
                        .flatten(start_dim=1),
                        "X=1",
                        step=1,
                        yLimits=[-5, 5],
                    )

                    Plotting.PlotFunctions(
                        axes[1, 2],
                        model(
                            4 * torch.ones_like(Xtrue),
                            iterations=ITERATIONS,
                            sample=True,
                        )
                        .detach()
                        .flatten(start_dim=1),
                        "X=4",
                        step=1,
                        yLimits=[-5, 5],
                    )

                model.train()
                plt.tight_layout()
                plt.show()
                plt.gcf().canvas.draw()
                plt.gcf().canvas.start_event_loop(1e-1)
