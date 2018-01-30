from collections import Counter
from langdetect import detect

def lang_detect(lines):
    l = []
    i = 0
    for line in lines:
        l.append(detect(line))
    lang1 = Counter(l).most_common(1)[0][0]
    return lang1

data_dir = './'
filenames = ['corpus1.txt', 'corpus2.txt']

for filename in filenames:
    with open(data_dir + filename, 'r') as f:
        lines = f.readlines()[:1000]
        print(lang_detect(lines))

