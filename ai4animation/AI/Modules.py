# Copyright (c) Meta Platforms, Inc. and affiliates.
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch.nn.parameter import Parameter


# Layers
class LinearLayer(torch.nn.Module):
    def __init__(self, input_size, output_size, dropout, activation):
        super(LinearLayer, self).__init__()

        self.Dropout = dropout
        self.Activation = activation
        self.Layer = nn.Linear(input_size, output_size)

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


class LinearFiLMLayer(torch.nn.Module):
    def __init__(self, input_size, output_size, film_size, dropout, activation):
        super(LinearFiLMLayer, self).__init__()

        self.FiLM = FiLMLayer(input_size, film_size)
        self.Layer = LinearLayer(input_size, output_size, dropout, activation)

    def forward(self, z, film):
        z = self.FiLM(z, film)
        z = self.Layer(z)
        return z


class LinearExpertsLayer(torch.nn.Module):
    def __init__(self, experts, input_size, output_size, dropout, activation):
        super(LinearExpertsLayer, self).__init__()

        self.Experts = experts
        self.InputDim = input_size
        self.OutputDim = output_size
        self.Dropout = dropout
        self.Activation = activation
        self.W = self.weights([experts, input_size, output_size])
        self.b = self.bias([experts, 1, output_size])

    def forward(self, z, w=None):
        shape = z.shape

        z = z.reshape(-1, z.shape[-1])
        z = F.dropout(z, self.Dropout, training=self.training)
        z = z.matmul(self.W) + self.b
        if w is not None:
            w = w.reshape(-1, w.shape[-1])
            w = w.swapaxes(0, 1).unsqueeze(-1)
            z = torch.sum(w * z, axis=0)

        if self.Activation is not None:
            z = self.Activation(z)

        if w is None:
            if len(shape) == 2:
                z = z.reshape(self.Experts, shape[0], -1)
            if len(shape) == 3:
                z = z.reshape(self.Experts, shape[0], shape[1], -1)
        else:
            if len(shape) == 2:
                z = z.reshape(shape[0], -1)
            if len(shape) == 3:
                z = z.reshape(shape[0], shape[1], -1)

        return z

    def weights(self, shape):
        alpha_bound = np.sqrt(6.0 / np.prod(shape[-2:]))
        alpha = np.asarray(
            np.random.uniform(low=-alpha_bound, high=alpha_bound, size=shape),
            dtype=np.float32,
        )
        return Parameter(torch.from_numpy(alpha), requires_grad=True)

    def bias(self, shape):
        return Parameter(torch.zeros(shape, dtype=torch.float), requires_grad=True)


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

        # Compute Latent Vector
        z = mu
        if sigma is None:
            z = z + torch.randn_like(x) * std
        else:
            z = z + torch.randn_like(x) * sigma

        _mu = mu.reshape(-1, self.SamplesSize)
        _lv = lv.reshape(-1, self.SamplesSize)
        kld = torch.mean(-0.5 * torch.sum(1 + _lv - _mu**2 - _lv.exp(), dim=1), dim=0)

        return z, kld, (mu, lv, std)


# Encoders
class LinearEncoder(torch.nn.Module):
    def __init__(self, input_size, hidden_size, output_size, dropout):
        super(LinearEncoder, self).__init__()

        self.L1 = LinearLayer(input_size, hidden_size, dropout, F.elu)
        self.L2 = LinearLayer(hidden_size, hidden_size, dropout, F.elu)
        self.L3 = LinearLayer(hidden_size, output_size, dropout, None)

    def forward(self, z):
        z = self.L1(z)
        z = self.L2(z)
        z = self.L3(z)
        return z


class LinearFiLMEncoder(torch.nn.Module):
    def __init__(self, input_size, hidden_size, output_size, film_size, dropout):
        super(LinearFiLMEncoder, self).__init__()

        self.L1 = LinearFiLMLayer(input_size, hidden_size, film_size, dropout, F.elu)
        self.L2 = LinearFiLMLayer(hidden_size, hidden_size, film_size, dropout, F.elu)
        self.L3 = LinearFiLMLayer(hidden_size, output_size, film_size, dropout, None)

    def forward(self, z, film):
        z = self.L1(z, film)
        z = self.L2(z, film)
        z = self.L3(z, film)
        return z


class RegularizedLinearFiLMEncoder(torch.nn.Module):
    def __init__(
        self,
        input_size,
        hidden_size,
        output_size,
        regularization_size,
        film_size,
        dropout,
    ):
        super(RegularizedLinearFiLMEncoder, self).__init__()

        self.L1 = LinearFiLMLayer(input_size, hidden_size, film_size, dropout, F.elu)
        self.L2 = LinearFiLMLayer(hidden_size, hidden_size, film_size, dropout, F.elu)
        self.L3 = LinearFiLMLayer(hidden_size, output_size, film_size, dropout, None)
        self.R = LinearFiLMLayer(
            hidden_size, regularization_size, film_size, dropout, None
        )

    def forward(self, z, film):
        z = self.L1(z, film)
        z = self.L2(z, film)
        y = self.L3(z, film)
        r = self.R(z, film)
        if self.training:
            return y, r
        else:
            return y


class LinearExpertsEncoder(torch.nn.Module):
    def __init__(self, experts, input_size, hidden_size, output_size, dropout):
        super(LinearExpertsEncoder, self).__init__()

        self.L1 = LinearExpertsLayer(experts, input_size, hidden_size, dropout, F.elu)
        self.L2 = LinearExpertsLayer(experts, hidden_size, hidden_size, dropout, F.elu)
        self.L3 = LinearExpertsLayer(experts, hidden_size, output_size, dropout, None)

    def forward(self, z, w):
        z = self.L1(z, w)
        z = self.L2(z, w)
        z = self.L3(z, w)
        return z


class LinearGatedExpertsEncoder(torch.nn.Module):
    def __init__(
        self,
        experts,
        input_size,
        hidden_size,
        output_size,
        gating_size,
        gating_hidden,
        dropout,
    ):
        super(LinearGatedExpertsEncoder, self).__init__()

        self.G = LinearEncoder(gating_size, gating_hidden, experts, dropout)
        self.L = LinearExpertsEncoder(
            experts, input_size, hidden_size, output_size, dropout
        )

    def forward(self, z, g):
        w = self.G(g)
        w = F.softmax(w, dim=-1)
        return self.L(z, w)


class QuantizationLayer(nn.Module):
    """
    Improved version over vector quantiser, with the dynamic initialisation
    for these unoptimised "dead" points.
    num_embed: number of codebook entry
    embed_dim: dimensionality of codebook entry
    beta: weight for the commitment loss
    distance: distance for looking up the closest code
    anchor: anchor sampled methods
    first_batch: if true, the offline version of our model
    contras_loss: if true, use the contras_loss to further improve the performance
    """

    def __init__(
        self,
        num_embed,
        embed_dim,
        beta,
        distance="cos",
        anchor="probrandom",
        first_batch=False,
        contras_loss=False,
    ):
        super().__init__()

        self.num_embed = num_embed
        self.embed_dim = embed_dim
        self.beta = beta
        self.distance = distance
        self.anchor = anchor
        self.first_batch = first_batch
        self.contras_loss = contras_loss
        self.decay = 0.99
        self.init = False

        self.pool = FeaturePool(self.num_embed, self.embed_dim)
        self.embedding = nn.Embedding(self.num_embed, self.embed_dim)
        self.embedding.weight.data.uniform_(-1.0 / self.num_embed, 1.0 / self.num_embed)
        self.register_buffer("embed_prob", torch.zeros(self.num_embed))

    def forward(self, z, temp=None, rescale_logits=False, return_logits=False):
        assert temp is None or temp == 1.0, "Only for interface compatible with Gumbel"
        assert rescale_logits == False, "Only for interface compatible with Gumbel"
        assert return_logits == False, "Only for interface compatible with Gumbel"
        # reshape z -> (batch, height, width, channel) and flatten
        # z = rearrange(z, 'b c h w -> b h w c').contiguous()
        z_flattened = z.view(-1, self.embed_dim)

        # calculate the distance
        if self.distance == "l2":
            # l2 distances from z to embeddings e_j (z - e)^2 = z^2 + e^2 - 2 e * z
            d = (
                -torch.sum(z_flattened.detach() ** 2, dim=1, keepdim=True)
                - torch.sum(self.embedding.weight**2, dim=1)
                + 2
                * torch.einsum(
                    "bd, dn-> bn",
                    z_flattened.detach(),
                    rearrange(self.embedding.weight, "n d-> d n"),
                )
            )
        elif self.distance == "cos":
            # cosine distances from z to embeddings e_j
            normed_z_flattened = F.normalize(z_flattened, dim=1).detach()
            normed_codebook = F.normalize(self.embedding.weight, dim=1)
            d = torch.einsum(
                "bd,dn->bn",
                normed_z_flattened,
                rearrange(normed_codebook, "n d -> d n"),
            )

        # encoding
        sort_distance, indices = d.sort(dim=1)

        # look up the closest point for the indices
        encoding_indices = indices[:, -1]
        encodings = torch.zeros(
            encoding_indices.unsqueeze(1).shape[0], self.num_embed, device=z.device
        )
        encodings.scatter_(1, encoding_indices.unsqueeze(1), 1)

        # quantise and unflatten
        z_q = torch.matmul(encodings, self.embedding.weight).view(z.shape)
        # compute loss for embedding
        loss = self.beta * torch.mean((z_q.detach() - z) ** 2) + torch.mean(
            (z_q - z.detach()) ** 2
        )
        # preserve gradients
        z_q = z + (z_q - z).detach()
        # reshape back to match original input shape
        # z_q = rearrange(z_q, 'b h w c -> b c h w').contiguous()
        # count
        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))
        min_encodings = encodings

        # online clustered reinitialisation for unoptimized points
        if self.training:
            # calculate the average usage of code entries
            self.embed_prob.mul_(self.decay).add_(avg_probs, alpha=1 - self.decay)
            # running average updates
            if self.anchor in ["closest", "random", "probrandom"] and (not self.init):
                # closest sampling
                if self.anchor == "closest":
                    sort_distance, indices = d.sort(dim=0)
                    random_feat = z_flattened.detach()[indices[-1, :]]
                # feature pool based random sampling
                elif self.anchor == "random":
                    random_feat = self.pool.query(z_flattened.detach())
                # probabilitical based random sampling
                elif self.anchor == "probrandom":
                    norm_distance = F.softmax(d.t(), dim=1)
                    prob = torch.multinomial(norm_distance, num_samples=1).view(-1)
                    random_feat = z_flattened.detach()[prob]
                # decay parameter based on the average usage
                decay = (
                    torch.exp(
                        -(self.embed_prob * self.num_embed * 10) / (1 - self.decay)
                        - 1e-3
                    )
                    .unsqueeze(1)
                    .repeat(1, self.embed_dim)
                )
                self.embedding.weight.data = (
                    self.embedding.weight.data * (1 - decay) + random_feat * decay
                )
                if self.first_batch:
                    self.init = True
            # contrastive loss
            if self.contras_loss:
                sort_distance, indices = d.sort(dim=0)
                dis_pos = sort_distance[
                    -max(1, int(sort_distance.size(0) / self.num_embed)) :, :
                ].mean(dim=0, keepdim=True)
                dis_neg = sort_distance[: int(sort_distance.size(0) * 1 / 2), :]
                dis = torch.cat([dis_pos, dis_neg], dim=0).t() / 0.07
                contra_loss = F.cross_entropy(
                    dis,
                    torch.zeros((dis.size(0),), dtype=torch.long, device=dis.device),
                )
                loss += contra_loss

        return z_q, loss, (perplexity, min_encodings, encoding_indices)

        # # quantise and unflatten
        # z_q = self.embedding.weight[encoding_indices].reshape(z.shape)


class FeaturePool:
    """
    This class implements a feature buffer that stores previously encoded features

    This buffer enables us to initialize the codebook using a history of generated features
    rather than the ones produced by the latest encoders
    """

    def __init__(self, pool_size, dim=64):
        """
        Initialize the FeaturePool class

        Parameters:
            pool_size(int) -- the size of featue buffer
        """
        self.pool_size = pool_size
        if self.pool_size > 0:
            self.nums_features = 0
            self.features = (torch.rand((pool_size, dim)) * 2 - 1) / pool_size

    def query(self, features):
        """
        return features from the pool
        """
        self.features = self.features.to(features.device)
        if self.nums_features < self.pool_size:
            if (
                features.size(0) > self.pool_size
            ):  # if the batch size is large enough, directly update the whole codebook
                random_feat_id = torch.randint(
                    0, features.size(0), (int(self.pool_size),)
                )
                self.features = features[random_feat_id]
                self.nums_features = self.pool_size
            else:
                # if the mini-batch is not large nuough, just store it for the next update
                num = self.nums_features + features.size(0)
                self.features[self.nums_features : num] = features
                self.nums_features = num
        else:
            if features.size(0) > int(self.pool_size):
                random_feat_id = torch.randint(
                    0, features.size(0), (int(self.pool_size),)
                )
                self.features = features[random_feat_id]
            else:
                random_id = torch.randperm(self.pool_size)
                self.features[random_id[: features.size(0)]] = features

        return self.features
