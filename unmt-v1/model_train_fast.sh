#! /usr/bin/env bash

python3 train.py \
    -src_lang en \
    -tgt_lang ru \
    -train_src_mono ../data/corpus.tok.clean.tc.en \
    -train_tgt_mono ../data/corpus.tok.clean.tc.ru \
    -train_src_bi ../data/parallel.tok.tc.en \
    -train_tgt_bi ../data/parallel.tok.tc.ru \
    -src_to_tgt_dict ../models/en-ru.txt \
    -tgt_to_src_dict ../models/ru-en.txt \
    -layers 3 \
    -rnn_size 4 \
    -src_vocab_size 500 \
    -tgt_vocab_size 500 \
    -print_every 1 \
    -save_every 2 \
    -batch_size 4 \
    -discriminator_hidden_size 128 \
    -supervised_only 1 \
    -src_vocabulary ../data/src.pickle \
    -tgt_vocabulary ../data/tgt.pickle \
    -all_vocabulary ../data/all.pickle 