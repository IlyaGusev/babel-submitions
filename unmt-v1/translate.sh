#! /usr/bin/env bash

set -x

mosesdecoder=/model/mosesdecoder
muse=/model/MUSE
src=src
tgt=tgt
epochs=$1   # 1
layers=$2   # 1
rnn_size=$3 # 150

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

# Join setntenses to produce better embeddings
cat /model/parallel.tok.clean.$src /model/corpus.tok.clean.$src > /model/full.$src
cat /model/parallel.tok.clean.$tgt /model/corpus.tok.clean.$tgt > /model/full.$tgt

# Train truecaser
$mosesdecoder/scripts/recaser/train-truecaser.perl -corpus /model/full.$src -model /model/corpus-truecase-model.$src
$mosesdecoder/scripts/recaser/train-truecaser.perl -corpus /model/full.$tgt -model /model/corpus-truecase-model.$tgt

# Apply truecaser
$mosesdecoder/scripts/recaser/truecase.perl -model /model/corpus-truecase-model.$src < /model/parallel.tok.clean.$src > /model/parallel.tok.clean.tc.$src
$mosesdecoder/scripts/recaser/truecase.perl -model /model/corpus-truecase-model.$tgt < /model/parallel.tok.clean.$tgt > /model/parallel.tok.clean.tc.$tgt
$mosesdecoder/scripts/recaser/truecase.perl -model /model/corpus-truecase-model.$src < /model/corpus.tok.clean.$src > /model/corpus.tok.clean.tc.$src
$mosesdecoder/scripts/recaser/truecase.perl -model /model/corpus-truecase-model.$tgt < /model/corpus.tok.clean.$tgt > /model/corpus.tok.clean.tc.$tgt
$mosesdecoder/scripts/recaser/truecase.perl -model /model/corpus-truecase-model.$src < /model/input.tok.clean.$src > /model/input.tok.clean.tc.$src

$mosesdecoder/scripts/recaser/truecase.perl -model /model/corpus-truecase-model.$src < /model/full.$src > /model/full.tc.$src
$mosesdecoder/scripts/recaser/truecase.perl -model /model/corpus-truecase-model.$tgt < /model/full.$tgt > /model/full.tc.$tgt

# Run FastText
fasttext skipgram -input /model/full.tc.$src -minCount 3 -epoch 10 -loss ns -thread 16 -dim 300 -output /model/embedding.ft.$src
fasttext skipgram -input /model/full.tc.$tgt -minCount 3 -epoch 10 -loss ns -thread 16 -dim 300 -output /model/embedding.ft.$tgt

# Run MUSE
python3 $muse/unsupervised.py --src_lang $src --tgt_lang $tgt --src_emb /model/embedding.ft.$src --tgt_emb /model/embedding.ft.$tgt --dis_most_frequent 0
cp dumped/*/vectors-$src.txt /model/embeddings/embedding.mu.$src
cp dumped/*/vectors-$tgt.txt /model/embeddings/embedding.mu.$tgt

# Train model
python3 /model/model_train.py -src_lang $src \
    -tgt_lang $tgt \
    -train_src_mono /model/corpus.tok.clean.tc.$src \
    -train_tgt_mono /model/corpus.tok.clean.tc.$tgt \
    -train_src_bi /model/parallel.tok.clean.tc.$src \
    -train_tgt_bi /model/parallel.tok.clean.tc.$tgt \
    -layers 3 \
    -rnn_size 200 \
    -src_vocab_size 40000 \
    -tgt_vocab_size 40000 \
    -print_every 100 \
    -batch_size 64 \
    -src_embeddings /model/embeddings/embedding.mu.$src \
    -tgt_embeddings /model/embeddings/embedding.mu.$tgt \
    -discriminator_hidden_size 512 \
    -supervised-only True \
    -supervised_epochs 5

# Prediction
python3 /model/model_translate.py -src_lang $src \
    -tgt_lang $tgt \
    -train_src_mono /model/corpus.tok.clean.tc.$src \
    -train_tgt_mono /model/corpus.tok.clean.tc.$tgt  \
    -lang src -model model.pt \
    -input /model/input.tok.clean.tc.$src \
    -output /model/output.tok.clean.tc.$tgt \
    -discriminator_hidden_size 512 \
    -layers 3 \
    -rnn_size 200 \
    -src_vocab_size 40000 \
    -tgt_vocab_size 40000 \
    -src_embeddings /model/embeddings/embedding.mu.$src \
    -tgt_embeddings /model/embeddings/embedding.mu.$tgt

# Apply detruecaser
$mosesdecoder/scripts/recaser/detruecase.perl < /model/output.tok.clean.tc.$tgt > /model/output.tok.$tgt

# Detokenize
$mosesdecoder/scripts/tokenizer/detokenizer.perl -threads 8 < /model/output.tok.$tgt > /output/output.txt

echo "DONE"
