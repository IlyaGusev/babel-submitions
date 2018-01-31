#! /usr/bin/env bash

python /model/lang_detect.py --corpus1 /data/corpus1.txt --corpus2 /data/corpus2.txt

cat /data/input.txt > /output/output.txt
