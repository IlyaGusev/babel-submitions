#! /usr/bin/env bash

python3 train.py -src_lang en -tgt_lang ru -train_src_mono ../data/corpus.tok.clean.tc.en \
    -train_tgt_mono ../data/corpus.tok.clean.tc.ru -train_src_bi ../data/parallel.tok.tc.en \
    -train_tgt_bi ../data/parallel.tok.tc.ru -src_to_tgt_dict ../models/en-ru.txt \
    -tgt_to_src_dict ../models/ru-en.txt -layers 3 -rnn_size 200 -src_vocab_size 40000 -tgt_vocab_size 40000 \
    -print_every 1000 -batch_size 64 -src_embeddings ../models/wiki.multi.en.vec -tgt_embeddings ../models/wiki.multi.ru.vec \
    -discriminator_hidden_size 512