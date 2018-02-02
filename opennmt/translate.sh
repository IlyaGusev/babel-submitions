#! /usr/bin/env bash

set -x

opennmt=/model/OpenNMT
src=src
tgt=tgt
epochs=$1   # 13

# Train-Validation split
lines=`wc -l < /data/parallel.$src`
lines_train=`python3 -c "print(int($lines*.90))"`
lines_val=`python3 -c "print($lines-$lines_train)"`

# Tokenize data
head -n$lines_train /data/parallel.$src | th $opennmt/tools/tokenize.lua > /model/train.$src.txt
tail -n$lines_val   /data/parallel.$src | th $opennmt/tools/tokenize.lua > /model/valid.$src.txt
head -n$lines_train /data/parallel.$tgt | th $opennmt/tools/tokenize.lua > /model/train.$tgt.txt
tail -n$lines_val   /data/parallel.$tgt | th $opennmt/tools/tokenize.lua > /model/valid.$tgt.txt

# PreProcess
th $opennmt/preprocess.lua -train_src /model/train.$src.txt -train_tgt /model/train.$tgt.txt -valid_src /model/valid.$src.txt -valid_tgt /model/valid.$tgt.txt -save_data /model/pretrained

# OpenNMT
th $opennmt/train.lua -data data/pretrained.t7 -save_model /model/data/pretrained -gpuid 1

echo "DONE"
