#! /usr/bin/env bash

set -x

lines=`wc -l < /data/input.txt`
if [ $lines -lt 101 ]; then
  cat /data/input.txt > /output/output.txt
  exit 0
fi

langs=("et" "fa" "fi" "gu" "he" "hi" "hr" "hu" "id" "it")

l=`python /model/lang_detect.py --corpus /data/corpus2.txt`
(for e in "${langs[@]}"; do [[ "$e" == "$l" ]] && exit 0; done) && { #found
  echo $l
  exit -1
} || { #not found
  echo "DONE"
  cat /data/input.txt > /output/output.txt
}
