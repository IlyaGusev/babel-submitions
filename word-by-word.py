import numpy as np
import argparse
from collections import defaultdict
from lang_detect import lang_detect
def parse_args():
    parser = argparse.ArgumentParser(
        description="tramslates input file into other language word-by-word")
    parser.add_argument('--input', required=True)
    parser.add_argument('--corpus1', required=True)
    parser.add_argument('--corpus2', required=True)
    parser.add_argument('--output', required=True)
    return parser.parse_args()

def get_dictionary(src, tgt):
    lines = open("./vocabs/" + src + "-" + tgt + ".txt", "r").readlines()
    dictionary = defaultdict(list)
    for line in lines:
        word_src, word_tgt = line.split()
        dictionary[word_src].append(word_tgt)
    return dict(dictionary)

def translate_word(word, dictionary):
    if word == "":
        return ""
    translations = dictionary.get(word.lower(), [word])
    translation = np.random.choice(translations)
    if word[0].isupper():
        translation = translation[0].upper() + translation[1:]
    return translation

def translate(sentence, dict_to_en, dict_from_en):
    translate1 = [translate_word(word, dict_to_en) for word in sentence.split()]
    translate2 = [translate_word(word, dict_from_en) for word in translate1]
    return " ".join(translate2)


if __name__ == "__main__":
    args = parse_args()

    corpus1 = open(args.corpus1, "r").readlines()

    corpus2 = open(args.corpus2, "r").readlines()

    input_data = open(args.input, "r").readlines()

    output_file = open(args.output, "w")

    lang1 = lang_detect(corpus1[:100])

    lang2 = lang_detect(corpus2[:100])
    
    dict1 = get_dictionary(lang1, "en")
    dict2 = get_dictionary("en", lang2)


    print(lang1, lang2)
    print(len(corpus1), len(corpus2), len(input_data))
    output_data = []
    for line in input_data:
        output_data.append(translate(line, dict1, dict2))
    output_file.write('\n'.join(output_data))
    output_file.close()




