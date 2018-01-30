#! /usr/bin/env bash

set -x
set -e

OUTPUT_DIR="./data"
mosesdecoder=${OUTPUT_DIR}/mosesdecoder

SRC=en
TGT=ru

prefix=$OUTPUT_DIR/corpus.tok.clean
$mosesdecoder/scripts/recaser/truecase.perl -model $prefix-truecase-model.$SRC < $prefix.$SRC > $prefix.tc.$SRC
$mosesdecoder/scripts/recaser/truecase.perl -model $prefix-truecase-model.$TGT < $prefix.$TGT > $prefix.tc.$TGT

# TODO: run translation

$mosesdecoder/scripts/recaser/detruecase.perl
