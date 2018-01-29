from src.word_by_word import WordByWordModel, inflate_vocabularies

SRC_LANG = "en"
TGT_LANG = "ru"
SRC_TO_TGT_DICT_FILENAME = "models/" + SRC_LANG + "-" + TGT_LANG + ".txt"
TGT_TO_SRC_DICT_FILENAME = "models/" + TGT_LANG + "-" + SRC_LANG + ".txt"
SRC_FILENAME = "bleu-" + SRC_LANG + ".txt"

if __name__ == "__main__":
    src_vocabulary, tgt_vocabulary = \
        inflate_vocabularies(SRC_TO_TGT_DICT_FILENAME, TGT_TO_SRC_DICT_FILENAME, SRC_LANG, TGT_LANG)
    model = WordByWordModel(SRC_TO_TGT_DICT_FILENAME, TGT_TO_SRC_DICT_FILENAME, src_vocabulary, tgt_vocabulary)

    OUTPUT_FILENAME = "bleu-" + TGT_LANG + "-pred.txt"
    with open(SRC_FILENAME, "r", encoding='utf-8') as r:
        with open(OUTPUT_FILENAME, "w", encoding='utf-8') as w:
            for line in r:
                line = line.strip()
                translation = model.translate_src2tgt_sentence(line)
                w.write(" ".join(translation[:-1]) + "\n")