from typing import List, Tuple

import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F
import numpy as np

from .vae import ModelVAE, Reparametrized
from ..vae_runner import VaeDataset
from ..components import Component


class SeqVAELoss(torch.nn.Module):
    def __init__(self, batch_size):
        super(SeqVAELoss, self).__init__()
        self.batch_size = batch_size

    def forward(self, decoder_output, mu_q, logvar_q, y_true_s, anneal):
        # Calculate KL Divergence loss
        kld = torch.mean(torch.sum(0.5 * (-logvar_q + torch.exp(logvar_q) + mu_q ** 2 - 1), -1))

        # Calculate Likelihood
        dec_shape = decoder_output.shape  # [batch_size x seq_len x total_items] = [1 x seq_len x total_items]

        decoder_output = F.log_softmax(decoder_output, -1)
        num_ones = float(torch.sum(y_true_s[0, 0]))

        likelihood = torch.sum(
            -1.0 * y_true_s.view(dec_shape[0] * dec_shape[1], -1) * \
            decoder_output.view(dec_shape[0] * dec_shape[1], -1)
        ) / (float(self.batch_size) * num_ones)

        final = (anneal * kld) + (likelihood)

        return final


SeqOutputs = Tuple[Tensor, Tensor, Tensor]


class SeqVAE(torch.nn.Module):
    def __init__(self, decoder_dims: List[int], components: List[Component],
                 item_embed_size: int, rnn_size: int, total_items: int,
                 scalar_parametrization: bool, encoder_dims=None, dropout=0.5, batch_size=1) -> None:
        super().__init__()
        self.device = torch.device("cpu")
        self.components = nn.ModuleList(components)

        self.total_z_dim = sum(component.dim for component in components)
        for component in components:
            component.init_layers(decoder_dims[0], scalar_parametrization=scalar_parametrization)

        self.total_items = total_items
        self.rnn_size = rnn_size
        self.item_embed_size = item_embed_size
        self.item_embed = nn.Embedding(self.total_items, self.item_embed_size)
        self.gru = nn.GRU(self.item_embed_size, self.rnn_size, batch_first = True, num_layers = 1)
        if not encoder_dims:
            encoder_dims = decoder_dims[::-1]
            encoder_dims[0] = self.rnn_size
        self.criterion = SeqVAELoss(batch_size=batch_size)
        self.encoder_dims, self.decoder_dims = encoder_dims, decoder_dims  # encoder and decoder dims
        self.decoder_dims.insert(0, self.total_z_dim)
        # 1 hidden layer encoder
        self.en_layers = nn.ModuleList([nn.Linear(d_in, d_out) for
                                        d_in, d_out in zip(self.encoder_dims[:-1], self.encoder_dims[1:])])

        self.de_layers = nn.ModuleList([nn.Linear(d_in, d_out) for
                                        d_in, d_out in zip(self.decoder_dims[:-1], self.decoder_dims[1:])])

        self.drop = nn.Dropout(dropout)
        self.init_weights()

    def to(self, device: torch.device):
        self.device = device
        return super().to(device)


    def init_weights(self):
        for layer in self.en_layers:
            # Xavier Initialization for weights
            size = layer.weight.size()
            fan_out = size[0]
            fan_in = size[1]
            std = np.sqrt(2.0 / (fan_in + fan_out))
            layer.weight.data.normal_(0.0, std)

            # Normal Initialization for Biases
            layer.bias.data.normal_(0.0, 0.001)

        for layer in self.de_layers:
            # Xavier Initialization for weights
            size = layer.weight.size()
            fan_out = size[0]
            fan_in = size[1]
            std = np.sqrt(2.0 / (fan_in + fan_out))
            layer.weight.data.normal_(0.0, std)

            # Normal Initialization for Biases
            layer.bias.data.normal_(0.0, 0.001)

    def encode(self, x: Tensor) -> Tensor:
        assert len(x.shape) == 2
        seq_len, dim = x.shape

        x = F.normalize(x)
        x = self.drop(x)

        for i, layer in enumerate(self.en_layers):
            x = layer(x)
            x = torch.tanh(x)

        return x.view(seq_len, -1)

    def decode(self, concat_z: Tensor) -> Tensor:
        assert len(concat_z.shape) >= 2
        bs = concat_z.size(-2)

        for i, layer in enumerate(self.de_layers[:-1]):
            concat_z = layer(concat_z)
            concat_z = torch.tanh(concat_z)

        concat_z = self.de_layers[-1](concat_z)

        concat_z = concat_z.view(-1, bs, self.total_items)  # flatten
        return concat_z.squeeze(dim=0)  # in case we're not doing LL estimation

    def forward(self, x: Tensor) -> SeqOutputs:
        in_shape = x.shape
        # print('FORWARD shapes:')
        # print('x: ', list(x.shape), end=' -> ')
        x = self.item_embed(x)
        # print('item_embed: ', list(x.shape), end=' -> ')
        rnn_out, _ = self.gru(x)
        # print('rnn_out: ', list(rnn_out.shape), end=' -> ')
        rnn_out = rnn_out.view(in_shape[0] * in_shape[1], -1)
        # print('rnn_out reshaped: ', list(rnn_out.shape), end=' -> ')
        x_encoded = self.encode(rnn_out)
        # print('encoded: ', list(x_encoded.shape))

        reparametrized = []
        z_mean = None
        z_log_sigma = None
        for component in self.components:
            q_z, p_z, z_params = component(x_encoded)
            z, data = q_z.rsample_with_parts()
            reparametrized.append(Reparametrized(q_z, p_z, z, data))
            # keep only the last latent variable
            z_mean = z_params[0]
            z_log_sigma = z_params[1]

        concat_z = torch.cat(tuple(x.z for x in reparametrized), dim=-1)
        x_ = self.decode(concat_z)
        x_ = x_.view(in_shape[0], in_shape[1], -1)
        return x_, z_mean, z_log_sigma

    def train_step(self, optimizer: torch.optim.Optimizer, x: Tensor, y_s: Tensor, beta: float):
        optimizer.zero_grad()

        x = x.to(self.device)
        x_, z_mean, z_log_sigma = self(x)
        loss = self.criterion(x_, z_mean, z_log_sigma, y_s, beta)

        assert torch.isfinite(loss).all()
        loss.backward()
        c_params = [v for k, v in self.named_parameters() if "curvature" in k]
        if c_params:  # TODO: Look into this, possibly disable it.
            torch.nn.utils.clip_grad_norm_(c_params, max_norm=1.0, norm_type=2)  # Enable grad clip?
        optimizer.step()
        return loss.data
        # return batch_stats.convert_to_float(), (reparametrized, concat_z, x_mb_)
