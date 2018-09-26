#!/usr/bin/env python
# -*- coding:utf-8 -*-

import torch
from torch import nn
from .multi_layer import MultiDenseLayer

class LSTMEncoder(nn.Module):

    __pad_index = 0
    __batch_first = True

    def __init__(self, n_vocab, n_dim_embedding, n_dim_lstm_hidden, n_lstm_layer, bidirectional, highway=False, **kwargs):

        super(__class__, self).__init__()

        self._n_vocab = n_vocab
        self._n_dim_embedding = n_dim_embedding
        self._n_dim_lstm_hidden = n_dim_lstm_hidden
        self._n_lstm_layer = n_lstm_layer
        self._bidirectional = bidirectional
        self._highway = highway

        self._embed = nn.Embedding(num_embeddings=n_vocab, embedding_dim=n_dim_embedding, padding_idx=self.__pad_index)
        self._lstm = nn.LSTM(input_size=n_dim_embedding, hidden_size=n_dim_lstm_hidden, num_layers=n_lstm_layer,
                             batch_first=self.__batch_first, bidirectional=bidirectional)

    def _init_state(self, batch_size):
        # zero init
        n_dim = self._n_lstm_layer*2 if self._bidirectional else self._n_lstm_layer
        h_0 = torch.zeros(n_dim, batch_size, self._n_dim_embedding)
        c_0 = torch.zeros(n_dim, batch_size, self._n_dim_embedding)
        return h_0, c_0

    def forward(self, x_seq, seq_len):
        """

        :param x_seq: batch of sequence of index; torch.tensor((n_mb, n_seq_len_max), dtype=torch.long)
        :param seq_len: batch of sequence length; torch.tensor((n_mb,), dtype=torch.long)
        """
        batch_size, n_seq_len_max = x_seq.size()

        # (n_mb, n_seq_len_max, n_dim_embedding)
        embed = self._embed(x_seq)
        # ignore padded state
        v = nn.utils.rnn.pack_padded_sequence(embed, lengths=seq_len, batch_first=self.__batch_first)

        h_0_c_0 = self._init_state(batch_size=batch_size)
        h, _ = self._lstm(v, h_0_c_0)

        # undo the packing operation
        h, _ = nn.utils.rnn.pad_packed_sequence(h, batch_first=self.__batch_first, padding_value=0.)

        if self._highway:
            h = h + embed
            # h = torch.cat([h,embed], dim=-1)

        return h, seq_len


class GMMLSTMEncoder(LSTMEncoder):

    __pad_index = 0
    __batch_first = True

    def __init__(self, n_vocab, n_dim_embedding, n_dim_lstm_hidden, n_lstm_layer, bidirectional,
                 encoder_alpha: MultiDenseLayer, encoder_mu: MultiDenseLayer, encoder_sigma: MultiDenseLayer, highway=False, **kwargs):

        super(__class__, self).__init__(n_vocab, n_dim_embedding, n_dim_lstm_hidden, n_lstm_layer, bidirectional, highway, **kwargs)
        self._enc_alpha = encoder_alpha
        self._enc_mu = encoder_mu
        self._enc_sigma = encoder_sigma

    def _masked_softmax(self, x: torch.Tensor, mask: torch.Tensor, dim=1):
        masked_vec = x * mask.float()
        max_vec = torch.max(masked_vec, dim=dim, keepdim=True)[0]
        exps = torch.exp(masked_vec-max_vec)
        masked_exps = exps * mask.float()
        masked_sums = masked_exps.sum(dim, keepdim=True)
        zeros=(masked_sums == 0)
        masked_sums += zeros.float()
        return masked_exps/masked_sums

    def forward(self, x_seq, seq_len):

        mask = (x_seq > self.__pad_index)
        h, seq_len = super(__class__, self).forward(x_seq, seq_len)

        # alpha = (N_b, N_t)
        # alpha[b,:] = softmax(MLP(h[b,:]))
        z_alpha = self._enc_alpha(h)
        z_alpha = z_alpha.squeeze(dim=-1)
        alpha = self._masked_softmax(z_alpha, mask, dim=1)

        # mu = (N_b, N_t, N_d)
        # mu[b,t] = MLP(h[b,t])
        z_mu = self._enc_mu(h)
        mu = z_mu * mask.float().unsqueeze(dim=-1)

        # sigma = (N_b, N_t)
        # sigma[b,t] = exp(MLP(h[b,t]))
        z_sigma = self._enc_sigma(h)
        z_sigma = z_sigma.squeeze(dim=-1)
        sigma = torch.exp(z_sigma) * mask.float()

        return alpha, mu, sigma
