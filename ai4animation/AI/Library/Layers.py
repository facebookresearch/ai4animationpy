# Copyright (c) Meta Platforms, Inc. and affiliates.
import torch
import torch.nn as nn
import torch.nn.functional as F
from ai4animation.AI.Library import Defaults, Manifolds


class LinearLayer(torch.nn.Module):
    def __init__(
        self,
        input_size,
        output_size,
        dropout=Defaults.Dropout,
        activation=Defaults.Activation,
    ):
        super(LinearLayer, self).__init__()

        self.InputSize = input_size
        self.OutputSize = output_size
        self.Dropout = dropout
        self.Activation = activation
        self.Layer = nn.Linear(input_size, output_size)

    def input_dim(self):
        return self.InputSize

    def output_dim(self):
        return self.OutputSize

    def forward(self, z):
        z = F.dropout(z, self.Dropout, training=self.training)
        z = self.Layer(z)
        if self.Activation is not None:
            z = self.Activation(z)
        return z


class FiLMLayer(torch.nn.Module):
    def __init__(self, feature_size, film_size):
        super(FiLMLayer, self).__init__()

        self.Scale = nn.Linear(film_size, feature_size)
        self.Shift = nn.Linear(film_size, feature_size)

    def forward(self, z, film):
        return self.Scale(film) * z + self.Shift(film)


class FiLMLinearLayer(torch.nn.Module):
    def __init__(
        self,
        input_size,
        output_size,
        film_size,
        dropout=Defaults.Dropout,
        activation=Defaults.Activation,
    ):
        super(FiLMLinearLayer, self).__init__()

        self.FiLM = FiLMLayer(input_size, film_size)
        self.Linear = LinearLayer(input_size, output_size, dropout, activation)

    def input_dim(self):
        return self.Linear.input_dim()

    def output_dim(self):
        return self.Linear.output_dim()

    def forward(self, z, film):
        z = self.FiLM(z, film)
        z = self.Linear(z)
        return z


class VariationalLayer(torch.nn.Module):
    def __init__(self, samples_size):
        super(VariationalLayer, self).__init__()

        self.SamplesSize = samples_size

        self.Mu = nn.Linear(samples_size, samples_size)
        self.LogVar = nn.Linear(samples_size, samples_size)

    def forward(self, x, sigma=None):
        mu = self.Mu(x)
        lv = self.LogVar(x)
        std = torch.exp(0.5 * lv)

        z = mu
        if sigma is None:
            z = z + torch.randn_like(x) * std
        else:
            z = z + torch.randn_like(x) * sigma

        _mu = mu.reshape(-1, self.SamplesSize)
        _lv = lv.reshape(-1, self.SamplesSize)
        kld = torch.mean(-0.5 * torch.sum(1 + _lv - _mu**2 - _lv.exp(), dim=1), dim=0)

        return z, kld, (mu, lv, std)


class CodebookLayer(torch.nn.Module):
    def __init__(self, channels, classes):
        super(CodebookLayer, self).__init__()

        self.Channels = channels
        self.Classes = classes

    def dimensions(self):
        return self.Channels * self.Classes

    def forward(self, z, sample):
        if sample is True:
            return Manifolds.Gumbel(z, self.Classes)
        if sample is False:
            return Manifolds.Softmax(z, self.Classes)
        return Manifolds.Gumbel(z, self.Classes, noise=sample)
