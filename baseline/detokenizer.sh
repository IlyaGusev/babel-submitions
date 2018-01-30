#! /usr/bin/env bash

set -x
set -e


# Tokenize data
./mosesdecoder/scripts/tokenizer/detokenizer.perl  -threads 8 < /output/output.tok.txt > /output/output.txt
