#! /usr/bin/env bash

set -x

lines=`wc -l < /data/input.txt`
if [ $lines -lt 101 ]; then
  cat /data/input.txt > /output/output.txt
  exit 0
fi

langs=("pt" "ro" "ru" "sk" "sl" "so" "sq" "sv" "sw" "ta")

l=`python /model/lang_detect.py --corpus /data/corpus2.txt`
(for e in "${langs[@]}"; do [[ "$e" == "$l" ]] && exit 0; done) && { #found
  echo $l
  exit -1
} || { #not found
  echo "DONE"
  cat /data/input.txt > /output/output.txt
}
