#! /usr/bin/env bash

set -x
set -e

OUTPUT_DIR="./data"
mkdir -p $OUTPUT_DIR

# Clone Moses
if [ ! -d "${OUTPUT_DIR}/mosesdecoder" ]; then
  echo "Cloning moses for data processing"
  git clone https://github.com/moses-smt/mosesdecoder.git "${OUTPUT_DIR}/mosesdecoder"
fi

mosesdecoder=${OUTPUT_DIR}/mosesdecoder

langs=(en ru)

# Tokenize data
for l in ${langs[@]}; do
  f="$OUTPUT_DIR/corpus.$l"
  echo "Tokenizing $f..."
  $mosesdecoder/scripts/tokenizer/tokenizer.perl -q -l $l -threads 8 < $f > ${f%.*}.tok.$l
done

function join_by { local IFS="$1"; shift; echo "$*"; }

# Clean all corpora
for f in ${OUTPUT_DIR}/*.tok.${langs}; do
  fbase=${f%.*}
  echo "Cleaning ${fbase}..."
  l=$(join_by " " ${langs[@]})
  ./clean-corpus-n-monolingual.perl $fbase $l "${fbase}.clean" 1 80
done

# Train truecaser
for f in ${OUTPUT_DIR}/*.tok.clean.${langs}; do
  fbase=${f%.*}
  echo "truecaser ${fbase}..."
  for l in ${langs[@]}; do
    $mosesdecoder/scripts/recaser/train-truecaser.perl -corpus $fbase.$l -model $fbase-truecase-model.$l
  done
done

echo "All done."
