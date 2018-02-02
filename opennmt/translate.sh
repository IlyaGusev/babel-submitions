#! /usr/bin/env bash

set -x

opennmt=/model/OpenNMT
src=src
tgt=tgt
epochs=$1   # 13

cut -f 1 /data/parallel_corpus.txt > /model/parallel.$src
cut -f 2 /data/parallel_corpus.txt > /model/parallel.$tgt

# Train-Validation split
lines=`wc -l < /model/parallel.$src`
lines_train=`python3 -c "print(int($lines*.90))"`
lines_val=`python3 -c "print($lines-$lines_train)"`

# Tokenize data
head -n$lines_train /model/parallel.$src | th $opennmt/tools/tokenize.lua > /model/train.$src.txt
tail -n$lines_val   /model/parallel.$src | th $opennmt/tools/tokenize.lua > /model/valid.$src.txt
head -n$lines_train /model/parallel.$tgt | th $opennmt/tools/tokenize.lua > /model/train.$tgt.txt
tail -n$lines_val   /model/parallel.$tgt | th $opennmt/tools/tokenize.lua > /model/valid.$tgt.txt

# PreProcess
th $opennmt/preprocess.lua -train_src /model/train.$src.txt -train_tgt /model/train.$tgt.txt -valid_src /model/valid.$src.txt -valid_tgt /model/valid.$tgt.txt -save_data /model/pretrained

# Train
th $opennmt/train.lua -optim adam -learning_rate 0.0002 -end_epoch $epochs -data /model/pretrained-train.t7 -save_model /model/prepared

# Translate
cat /data/input.txt | th $opennmt/tools/tokenize.lua > /model/input.txt

th $opennmt/translate.lua -model /model/prepared_epoch${epochs}_*.t7 -src /model/input.txt -output /model/output.txt

cat /model/output.txt | th $opennmt/tools/detokenize.lua > /output/output.txt

echo "DONE"

