#!/usr/bin/env python
# -*- coding:utf-8 -*-

import os, sys, io
import argparse
from contextlib import ExitStack
from typing import List, Dict, Union, Any

import numpy as np
import pickle
import progressbar

import torch

wd = os.path.dirname(__file__)
wd = "." if wd == "" else wd
os.chdir(wd)

from common.loader.text import TextLoader
from preprocess.tokenizer import CharacterTokenizer, AbstractTokenizer
from preprocess.corpora import Dictionary
from preprocess.dataset_feeder import SeqToGMMFeeder
from preprocess import utils

from model.multi_layer import MultiDenseLayer
from model.encoder import GMMLSTMEncoder
from model.loss import GMMSlicedWassersteinDistance_Parallel as WassersteinDistance
from model.loss import MaskedKLDivLoss
from model.utils import manual_backward

def config_corpus(project_dir: str = "./") -> Dict[str,str]:
    corpus_name = "d-n-s-min_max_2-(3,4,250)-100000-5-10"
    path_corpus = {
        "sentence":os.path.join(project_dir, "./toy_dataset/" + corpus_name + "_sentence.txt"),
        "gaussian_mixture":os.path.join(project_dir, "./toy_dataset/" + corpus_name + "_gmm.pickle"),
        "dictionary":os.path.join(project_dir, "./toy_dataset/" + corpus_name + "_sentence.pickle"),
        "trained_model_path":os.path.join(project_dir, "./saved_model/" + corpus_name + ".model"),
        "log_file_path":os.path.join(project_dir, "log_train_progress.log"),
        "corpus_size":100000
    }
    return path_corpus


def config_train(dictionary: Dictionary) -> Dict[str, Any]:
    n_dim_gmm = 2
    highway = False
    bidirectional = True
    n_dim_lstm_hidden = 40
    n_dim_embedding = 40
    n_batch = 100
    n_dim_h = n_dim_lstm_hidden * (bidirectional+1) + highway * n_dim_embedding
    cfg_encoder = {
        "lstm": {
        "n_vocab": dictionary.n_vocab + dictionary.offset,
        "n_dim_embedding":n_dim_embedding,
        "n_dim_lstm_hidden":n_dim_lstm_hidden,
        "n_lstm_layer":1,
        "bidirectional":bidirectional,
        "highway":highway
        },
        "alpha": {
            "n_dim_in":n_dim_h,
            "n_dim_out":1,
            "n_dim_hidden":10,
            "n_hidden":3,
            "activation_function":torch.relu
        },
        "mu": {
            "n_dim_in":n_dim_h,
            "n_dim_out":n_dim_gmm,
            "n_dim_hidden":10,
            "n_hidden":3,
            "activation_function":torch.relu
        },
        "sigma": {
            "n_dim_in":n_dim_h,
            "n_dim_out":1,
            "n_dim_hidden":10,
            "n_hidden":3,
            "activation_function":torch.relu
        },
        "wasserstein": {
            "n_dim":n_dim_gmm,
            "n_slice":10,
            "n_integral_point":100,
            "inv_cdf_method":"empirical",
            "scale_gradient":True
        },
        "regularize":True,
        "optim": {
            "n_epoch":30,
            "n_batch":n_batch,
            "optimizer":torch.optim.Adam,
            "lr":0.01,
            "validation_split":0.1
        }
    }
    return cfg_encoder


def _parse_args():

    parser = argparse.ArgumentParser(description="Sequence-to-GMM Encoder: train/validation script for toy dataset")
    parser.add_argument("--verbose", action="store_true", help="output verbosity")
    args = parser.parse_args()

    return args


def main_minibatch(sentence_encoder, optimizer, loss_function, reg_loss_function, token, seq_len, *tup_param_y, train_mode: bool = True):

    if train_mode:
        optimizer.zero_grad()

    with ExitStack() as context_stack:
        # if not train mode, enter into no_grad() context
        if not train_mode:
            context_stack.enter_context(torch.no_grad())

        if reg_loss_function is not None:
            lst_arr_unif = [np.full(n, 1./n, dtype=np.float32) for n in seq_len]
            arr_unif = utils.pad_numpy_sequence(lst_arr_unif)
            v_alpha_unif = torch.from_numpy(arr_unif)

        #### 1. forward coputation
        v_token = torch.tensor(token, dtype=torch.long)
        v_seq_len = torch.tensor(seq_len, dtype=torch.long)
        v_mask = (v_token > 0).float()

        tup_v_param_x = sentence_encoder(v_token, v_seq_len)
        v_alpha = tup_v_param_x[0]

        #### regularizer: kl-divergence
        if reg_loss_function is not None:
            reg_loss = reg_loss_function(v_alpha, v_alpha_unif, v_mask)
        else:
            reg_loss = 0.0

        #### 2. manual backpropagation

        # 2-1. convert from torch.Tensor to packed numpy array
        tup_param_x = tuple(map(lambda x:utils.pack_padded_sequence(x, lst_seq_len=seq_len), tup_v_param_x))

        # 2-2. Calculate batch-wise Sliced Wasserstein Distance and its gradients
        ## predicted parameters
        iter_gmm_param_x = zip(*tup_param_x)
        ## ground-truth parameters
        iter_gmm_param_y = zip(*tup_param_y)
        ## wasserstein distance and gradients
        dist, *tup_grad_param_x = \
            loss_function.sliced_wasserstein_distance_batch(iter_gmm_param_x, iter_gmm_param_y)

    #### 3. update model parameters
    if train_mode:
        # 3-1. pad list of gradient arrays
        tup_grad_param_x = tuple(map(utils.pad_numpy_sequence, tup_grad_param_x))
        # 3-2. call pytorch backward() operation
        if reg_loss_function is not None:
            reg_loss.backward(retain_graph=True)
        manual_backward(lst_tensor=tup_v_param_x, lst_gradient=tup_grad_param_x)

        ### update parameters
        optimizer.step()

    prefix = "train" if train_mode else "val"
    metrics = {
        f"{prefix}_dist":float(np.mean(dist)),
        f"{prefix}_kldiv":float(reg_loss),
        f"{prefix}_max_alpha":float(np.max(v_alpha.data.numpy()))
    }

    return metrics


def main():

    args = _parse_args()

    # setup corpus
    cfg_corpus = config_corpus()
    print("corpus:")
    for k, v in cfg_corpus.items():
        print(f"\t{k}:{v}")
    path_trained_model = cfg_corpus["trained_model_path"]
    corpus = TextLoader(file_path=cfg_corpus["sentence"])
    tokenizer = CharacterTokenizer()
    dictionary = Dictionary.load(file_path=cfg_corpus["dictionary"])
    with io.open(cfg_corpus["gaussian_mixture"], mode="rb") as ifs:
        dict_lst_gmm = pickle.load(ifs)

    # setup logger
    path_log_file = cfg_corpus["log_file_path"]
    if os.path.exists(path_log_file):
        os.remove(path_log_file)
    logger = io.open(path_log_file, mode="w")

    # setup sentence-to-gmm encoder
    cfg_train = config_train(dictionary=dictionary)

    for param_name in "alpha,mu,sigma".split(","):
        cfg_train["lstm"][f"encoder_{param_name}"] = MultiDenseLayer(**cfg_train[param_name])
    sentence_encoder = GMMLSTMEncoder(**cfg_train["lstm"])
    loss_function = WassersteinDistance(**cfg_train["wasserstein"])
    if cfg_train["regularize"]:
        reg_loss_function = MaskedKLDivLoss(reduction="sum")
    else:
        reg_loss_function = None

    # setup optimizer
    optimizer = cfg_train["optim"]["optimizer"](sentence_encoder.parameters(), lr=cfg_train["optim"]["lr"])

    # instanciate dataset feeder
    dataset_feeder = SeqToGMMFeeder(corpus=corpus, tokenizer=tokenizer, dictionary=dictionary,
                                     dict_lst_gmm_param=dict_lst_gmm, n_minibatch=cfg_train["optim"]["n_batch"],
                                     validation_split=cfg_train["optim"]["validation_split"],
                                     convert_var_to_std=True)

    # start training
    n_epoch = cfg_train["optim"]["n_epoch"]
    # iterate over epoch
    for n_epoch in range(n_epoch):
        print(f"epoch:{n_epoch}")

        n_processed = 0
        q = progressbar.ProgressBar(max_value=cfg_corpus["corpus_size"])
        q.update(n_processed)

        # iterate over mini-batch
        ## notice: tup_param_* = (lst_alpha[batch], lst_mu[batch, lst_sigma[batch)
        for train, validation in dataset_feeder:
            metrics = {
                "epoch":n_epoch,
                "processed":n_processed
            }

            # training
            token, *tup_param_y = map(list, zip(*train))
            seq_len, token, *tup_param_y = utils.len_pad_sort(token, *tup_param_y)
            metrics_train = main_minibatch(sentence_encoder, optimizer, loss_function, reg_loss_function, token, seq_len, *tup_param_y, train_mode=True)
            metrics.update(metrics_train)
            n_processed += len(seq_len)

            # validation
            token, *tup_param_y = map(list, zip(*validation))
            seq_len, token, *tup_param_y = utils.len_pad_sort(token, *tup_param_y)
            metrics_val = main_minibatch(sentence_encoder, optimizer, loss_function, reg_loss_function, token, seq_len, *tup_param_y, train_mode=False)
            metrics.update(metrics_val)
            n_processed += len(seq_len)

            # logging
            sep = "\t"
            f_value_to_str = lambda v: f"{v:1.7f}" if isinstance(v,float) else f"{v}"
            if n_epoch == 0 and metrics["processed"] == 0:
                s_header = sep.join(metrics.keys()) + "\n"
                logger.write(s_header)
            s_record = sep.join( map(f_value_to_str, metrics.values()) ) + "\n"
            logger.write(s_record)

            if args.verbose:
                s_print = ", ".join( [f"{k}:{v}" for k, v in zip(s_header.split(sep), s_record.split(sep))] )
                print(s_print)

            # next iteration
            q.update(n_processed)

        # save progress
        path_trained_model_e = path_trained_model + "." + str(n_epoch)
        print(f"saving...:{path_trained_model_e}")
        torch.save(sentence_encoder.state_dict(), path_trained_model_e)


if __name__ == "__main__":
    main()
    print("finished. good-bye.")