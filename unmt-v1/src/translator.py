import torch
from torch.autograd import Variable

from utils.batch import indices_from_sentence
from utils.vocabulary import Vocabulary


class Translator:
    @staticmethod
    def translate(model, sentence, src_lang, tgt_lang, vocabulary, use_cuda):
        translator = model.translate_to_tgt if tgt_lang == "tgt" else model.translate_to_src
        variable, lengths = Translator.sentence_to_variable(sentence, src_lang, vocabulary, use_cuda)
        translated = list(translator(variable, lengths)[:, 0])
        words = []
        for i in translated:
            index = i.data[0]
            word = vocabulary.get_word(index)[4:]
            if word == "</s>" or word == "<pad>":
                break
            words.append(word)
        return " ".join(words)

    @staticmethod
    def sentence_to_variable(sentence, lang, vocabulary, use_cuda):
        indices = indices_from_sentence(sentence, vocabulary, lang)
        variable = Variable(torch.zeros(1, len(indices))).type(torch.LongTensor)
        indices = Variable(torch.LongTensor(indices))
        variable[0] = indices
        variable = variable.transpose(0, 1)
        variable = variable.cuda() if use_cuda else variable
        lengths = [len(indices)]
        return variable, lengths
