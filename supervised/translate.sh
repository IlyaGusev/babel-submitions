#! /usr/bin/env bash

mosesdecoder=/model/mosesdecoder
src=src
tgt=tgt

# Randomize supervised input
shuf /data/parallel_corpus.txt > /model/parallel_corpus_shuffled.txt

# Split parallel corpus
cut -f 1 /model/parallel_corpus_shuffled.txt > /model/parallel.$src
cut -f 2 /model/parallel_corpus_shuffled.txt > /model/parallel.$tgt

# Tokenize data
$mosesdecoder/scripts/tokenizer/tokenizer.perl -q -threads 16 < /model/parallel.$src > /model/parallel.tok.$src
$mosesdecoder/scripts/tokenizer/tokenizer.perl -q -threads 16 < /model/parallel.$tgt > /model/parallel.tok.$tgt
$mosesdecoder/scripts/tokenizer/tokenizer.perl -q -threads 16 < /data/corpus1.txt > /model/corpus.tok.$src
$mosesdecoder/scripts/tokenizer/tokenizer.perl -q -threads 16 < /data/corpus2.txt > /model/corpus.tok.$tgt
$mosesdecoder/scripts/tokenizer/tokenizer.perl -q -threads 16 < /data/input.txt > /model/input.tok.$src

# Clean all corpora
$mosesdecoder/scripts/training/clean-corpus-n.perl /model/parallel.tok $src $tgt /model/parallel.tok.clean 1 80
/model/clean-corpus-monolingual.perl /model/corpus.tok.$src /model/corpus.tok.clean.$src 1 80
/model/clean-corpus-monolingual.perl /model/corpus.tok.$tgt /model/corpus.tok.clean.$tgt 1 80
/model/clean-corpus-monolingual.perl /model/input.tok.$src  /model/input.tok.clean.$src 1 80

# Train truecaser
$mosesdecoder/scripts/recaser/train-truecaser.perl -corpus /model/corpus.tok.clean.$src -model /model/corpus-truecase-model.$src
$mosesdecoder/scripts/recaser/train-truecaser.perl -corpus /model/corpus.tok.clean.$tgt -model /model/corpus-truecase-model.$tgt

# Apply truecaser
$mosesdecoder/scripts/recaser/truecase.perl -model /model/corpus-truecase-model.$src < /model/parallel.tok.clean.$src > /model/parallel.tok.clean.tc.$src
$mosesdecoder/scripts/recaser/truecase.perl -model /model/corpus-truecase-model.$tgt < /model/parallel.tok.clean.$tgt > /model/parallel.tok.clean.tc.$tgt
$mosesdecoder/scripts/recaser/truecase.perl -model /model/corpus-truecase-model.$src < /model/corpus.tok.clean.$src > /model/corpus.tok.clean.tc.$src
$mosesdecoder/scripts/recaser/truecase.perl -model /model/corpus-truecase-model.$tgt < /model/corpus.tok.clean.$tgt > /model/corpus.tok.clean.tc.$tgt
$mosesdecoder/scripts/recaser/truecase.perl -model /model/corpus-truecase-model.$src < /model/input.tok.clean.$src > /model/input.tok.clean.tc.$src

# TODO: remove
cp /model/input.tok.clean.tc.$src /model/output.tok.clean.tc.$tgt

# Apply detruecaser
$mosesdecoder/scripts/recaser/detruecase.perl < /model/output.tok.clean.tc.$tgt > /model/output.tok.$tgt

# Detokenize
$mosesdecoder/scripts/tokenizer/detokenizer.perl -threads 8 < /model/output.tok.$tgt > /output/output.txt

echo "DONE"
