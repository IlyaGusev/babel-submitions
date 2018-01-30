#! /usr/bin/env bash

set -x
set -e


# Clone Moses
if [ ! -d "./mosesdecoder" ]; then
  echo "Cloning moses for data processing"
  git clone https://github.com/moses-smt/mosesdecoder.git --depth 1 "./mosesdecoder"
fi

# Tokenize data
./mosesdecoder/scripts/tokenizer/tokenizer.perl -q  -threads 8 < ./data/corpus1.txt > ./data/corpus1.tok.txt
./mosesdecoder/scripts/tokenizer/tokenizer.perl -q  -threads 8 < ./data/corpus2.txt > ./data/corpus2.tok.txt
./mosesdecoder/scripts/tokenizer/tokenizer.perl -q  -threads 8 < ./data/input.txt > ./data/input.tok.txt
echo "All done."
