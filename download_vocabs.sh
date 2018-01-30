#! /usr/bin/env bash

set -x
set -e

OUTPUT_DIR="./vocabs"
mkdir -p $OUTPUT_DIR

for p in de ru af bs hr fr ja mk pl sk ta vi sq bg cs et de hu ko ms pt sl th ar ca da tl el id lv no ro es tr bn zh nl fi he it lt fa ru sv uk ; do
  src=en
  tgt=$p
  wget -nc -nv --show-progress -P ${OUTPUT_DIR} https://s3.amazonaws.com/arrival/dictionaries/$src-$tgt.txt
  wget -nc -nv --show-progress -P ${OUTPUT_DIR} https://s3.amazonaws.com/arrival/dictionaries/$tgt-$src.txt
done

wget -nc -nv --show-progress -P ${OUTPUT_DIR} https://s3.amazonaws.com/arrival/dictionaries/en-en.txt

