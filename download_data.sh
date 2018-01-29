#! /usr/bin/env bash

set -x
set -e

OUTPUT_DIR="./models"
mkdir -p $OUTPUT_DIR

for p in en,de en,ru; do IFS=",";
  set $p
  src=$1
  tgt=$2
  wget -nc -nv --show-progress -P ${OUTPUT_DIR} https://s3.amazonaws.com/arrival/dictionaries/$src-$tgt.txt
  wget -nc -nv --show-progress -P ${OUTPUT_DIR} https://s3.amazonaws.com/arrival/dictionaries/$tgt-$src.txt
  wget -nc -nv --show-progress -P ${OUTPUT_DIR} https://s3.amazonaws.com/arrival/embeddings/wiki.multi.$src.vec
  wget -nc -nv --show-progress -P ${OUTPUT_DIR} https://s3.amazonaws.com/arrival/embeddings/wiki.multi.$tgt.vec
done

DATA_DIR="./data"

mkdir -p ${DATA_DIR}
wget https://www.dropbox.com/s/b72usjc4fakhidc/En-Ru.zip -O ${DATA_DIR}/data.zip
unzip ${DATA_DIR}/data.zip -d ${DATA_DIR}
mv ${DATA_DIR}/corpus1.txt ${DATA_DIR}/corpus.en
mv ${DATA_DIR}/corpus2.txt ${DATA_DIR}/corpus.ru
