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
    -rnn_size 400 \
    -src_vocab_size 50000 \
    -tgt_vocab_size 50000 \
    -print_every 100 \
    -save_every 1000 \
    -batch_size 64 \
    -src_embeddings ../models/wiki.multi.en.vec \
    -tgt_embeddings ../models/wiki.multi.ru.vec \
    -discriminator_hidden_size 1024 \
    -supervised_only 1 \
    -src_vocabulary ../data/src.pickle \
    -tgt_vocabulary ../data/tgt.pickle \
    -all_vocabulary ../data/all.pickle 