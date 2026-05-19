# Copyright (c) Meta Platforms, Inc. and affiliates.
import torch
from ai4animation.AI.Library import Defaults
from ai4animation.AI.Library.Layers import FiLMLinearLayer, LinearLayer


class LinearBlock(torch.nn.Module):
    def __init__(
        self,
        input_size,
        output_size,
        hidden_size,
        dropout=Defaults.Dropout,
        activation=Defaults.Activation,
    ):
        super(LinearBlock, self).__init__()

        self.L1 = LinearLayer(input_size, hidden_size, dropout, activation)
        self.L2 = LinearLayer(hidden_size, hidden_size, dropout, activation)
        self.L3 = LinearLayer(hidden_size, output_size, dropout, None)

    def input_dim(self):
        return self.L1.input_dim()

    def output_dim(self):
        return self.L3.output_dim()

    def forward(self, z):
        z = self.L1(z)
        z = self.L2(z)
        z = self.L3(z)
        return z


class FiLMLinearBlock(torch.nn.Module):
    def __init__(
        self,
        input_size,
        output_size,
        hidden_size,
        film_size,
        dropout=Defaults.Dropout,
        activation=Defaults.Activation,
    ):
        super(FiLMLinearBlock, self).__init__()

        self.L1 = FiLMLinearLayer(
            input_size, hidden_size, film_size, dropout, activation
        )
        self.L2 = FiLMLinearLayer(
            hidden_size, hidden_size, film_size, dropout, activation
        )
        self.L3 = FiLMLinearLayer(hidden_size, output_size, film_size, dropout, None)

    def input_dim(self):
        return self.L1.input_dim()

    def output_dim(self):
        return self.L3.output_dim()

    def forward(self, z, film):
        z = self.L1(z, film)
        z = self.L2(z, film)
        z = self.L3(z, film)
        return z


class SpaceTimeBlock(torch.nn.Module):
    def __init__(
        self,
        input_dim,
        output_dim,
        hidden_dim,
        source_length,
        target_length,
        dropout,
    ):
        super(SpaceTimeBlock, self).__init__()
        self.Space = LinearBlock(input_dim, output_dim, hidden_dim, dropout)
        self.Time = LinearBlock(source_length, target_length, hidden_dim, dropout)

    def input_dim(self):
        return self.Space.input_dim()

    def output_dim(self):
        return self.Space.output_dim()

    def forward(self, z):
        z = self.Space(z)
        z = z.swapaxes(-1, -2)
        z = self.Time(z)
        z = z.swapaxes(-1, -2)
        return z


class RegularizedFiLMLinearBlock(torch.nn.Module):
    def __init__(
        self,
        input_size,
        output_size,
        regularization_size,
        hidden_size,
        film_size,
        dropout=Defaults.Dropout,
        activation=Defaults.Activation,
    ):
        super(RegularizedFiLMLinearBlock, self).__init__()

        self.L1 = FiLMLinearLayer(
            input_size, hidden_size, film_size, dropout, activation
        )
        self.L2 = FiLMLinearLayer(
            hidden_size, hidden_size, film_size, dropout, activation
        )
        self.L3 = FiLMLinearLayer(hidden_size, output_size, film_size, dropout, None)
        self.R = FiLMLinearLayer(
            hidden_size, regularization_size, film_size, dropout, None
        )

    def output_dim(self):
        return self.L3.output_dim()

    def regularization_dim(self):
        return self.R.output_dim()

    def forward(self, z, film):
        z = self.L1(z, film)
        z = self.L2(z, film)
        y = self.L3(z, film)
        if self.training:
            r = self.R(z, film)
            return y, r
        else:
            return y
