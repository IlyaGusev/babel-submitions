#! /usr/bin/env bash

set -x
set -e

OUTPUT_DIR="./models/"
mkdir -p $OUTPUT_DIR

wget https://www.dropbox.com/s/b72usjc4fakhidc/En-Ru.zip -O ${OUTPUT_DIR}/En-Ru.zip
unzip ${OUTPUT_DIR}/En-Ru.zip


for p in en,de en,ru; do IFS=",";
  set $p
  src=$1
  tgt=$2
  wget -nc -nv --show-progress -P ${OUTPUT_DIR} https://s3.amazonaws.com/arrival/dictionaries/$src-$tgt.txt
  wget -nc -nv --show-progress -P ${OUTPUT_DIR} https://s3.amazonaws.com/arrival/dictionaries/$tgt-$src.txt
  wget -nc -nv --show-progress -P ${OUTPUT_DIR} https://s3.amazonaws.com/arrival/embeddings/wiki.multi.$src.vec
  wget -nc -nv --show-progress -P ${OUTPUT_DIR} https://s3.amazonaws.com/arrival/embeddings/wiki.multi.$tgt.vec
done

TEXT_DIR="./data"

mkdir -p ${TEXT_DIR}
wget https://www.dropbox.com/s/b72usjc4fakhidc/En-Ru.zip data.zip
mv data.zip ${TEXT_DIR}/data.zip
unzip ${TEXT_DIR}/data.zip
mv ${TEXT_DIR}/corpus1.txt ${TEXT_DIR}/corpus.en
mv ${TEXT_DIR}/corpus2.txt ${TEXT_DIR}/corpus.ru