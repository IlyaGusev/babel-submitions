#! /usr/bin/env bash

set -x

lines=`wc -l < /data/input.txt`
if [ $lines -lt 101 ]; then
  cat /data/input.txt > /output/output.txt
  exit 0
fi

corpus=$1           # corpus2.txt
language=$2         # en

l=`python /model/lang_detect.py --corpus /data/$corpus`
if [[ "$l" == "$language" ]]; then
  echo $language
  exit -1
fi

echo "DONE"

cat /data/input.txt > /output/output.txt
