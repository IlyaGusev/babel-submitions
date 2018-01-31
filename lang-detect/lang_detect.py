import argparse
from collections import Counter
from langdetect import detect

def lang_detect(lines):
    l = []
    i = 0
    for line in lines:
        l.append(detect(line))
    lang1 = Counter(l).most_common(1)[0][0]
    return lang1

def parse_args():
    parser = argparse.ArgumentParser(
        description="detects languages")
    parser.add_argument('--corpus1', required=True)
    parser.add_argument('--corpus2', required=True)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    corpus1 = open(args.corpus1, "r").readlines()

    corpus2 = open(args.corpus2, "r").readlines()

    lang1 = lang_detect(corpus1[:1000])
    lang2 = lang_detect(corpus2[:1000])

    print(lang1, lang2)
