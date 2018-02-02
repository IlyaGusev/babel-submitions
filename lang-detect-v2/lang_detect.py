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
        description="detects language")
    parser.add_argument('--corpus', required=True)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    corpus = open(args.corpus, "r").readlines()

    lang = lang_detect(corpus[:1000])

    print(lang)
