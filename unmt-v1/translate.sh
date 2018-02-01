#! /usr/bin/env bash

python3 translate.py -src_lang en -tgt_lang ru -train_src_mono ../data/corpus.tok.clean.tc.en \
    -train_tgt_mono ../data/corpus.tok.clean.tc.ru -layers 3 -rnn_size 200 -src_vocab_size 40000 -tgt_vocab_size 40000 \
    -lang src -model model.pt -input ../data/input.tok.tc.txt -output ../data/pred.txt