#!/usr/bin/env python
# -*- coding:utf-8 -*-

import torch
from torch import nn

class LSTMEncoder(nn.Module):

    __pad_index = 0
    __batch_first = True

    def __init__(self, n_vocab, n_dim_embedding, n_dim_lstm_hidden, n_lstm_layer, bidirectional):

        self._n_vocab = n_vocab
        self._n_dim_embedding = n_dim_embedding
        self._n_dim_lstm_hidden = n_dim_lstm_hidden
        self._n_lstm_layer = n_lstm_layer
        self._bidirectional = bidirectional

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
        v = self._embed(x_seq)
        # ignore padded state
        v = nn.utils.rnn.pack_padded_sequence(v, lengths=seq_len, batch_first=self.__batch_first)

        h_0_c_0 = self._init_state(batch_size=batch_size)
        h, _ = self._lstm(v, h_0_c_0)

        # undo the packing operation
        h, _ = nn.utils.rnn.pad_packed_sequence(h, batch_first=self.__batch_first, padding_value=0.)

        return h, seq_len