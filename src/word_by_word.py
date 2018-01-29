from torch.autograd import Variable
import torch
from utils.vocabulary import Vocabulary
from typing import List, Tuple, Dict
from utils.batch import indices_from_sentence


def inflate_vocabularies(src_to_tgt_dict_filename: str, tgt_to_src_dict_filename: str, src_lang: str, tgt_lang: str):
    src_vocabulary = Vocabulary(language=src_lang)
    tgt_vocabulary = Vocabulary(language=tgt_lang)
    with open(src_to_tgt_dict_filename, "r", encoding='utf-8') as r:
        for line in r:
            src_word, tgt_word = line.strip().split()
            src_vocabulary.add_word(src_word)
            tgt_vocabulary.add_word(tgt_word)
    with open(tgt_to_src_dict_filename, "r", encoding='utf-8') as r:
        for line in r:
            tgt_word, src_word = line.strip().split()
            src_vocabulary.add_word(src_word)
            tgt_vocabulary.add_word(tgt_word)
    return src_vocabulary, tgt_vocabulary


class WordByWordModel:
    def __init__(self, src_to_tgt_dict_filename: str, tgt_to_src_dict_filename: str, src_vocabulary: Vocabulary,
                 tgt_vocabulary: Vocabulary, max_length: int=50):
        self.max_length = max_length
        self.src_to_tgt_dict_filename, self.tgt_to_src_dict_filename = \
            src_to_tgt_dict_filename, tgt_to_src_dict_filename
        self.src_vocabulary, self.tgt_vocabulary = src_vocabulary, tgt_vocabulary

        self.src2tgt = self.init_mapping(src_to_tgt_dict_filename, self.src_vocabulary, self.tgt_vocabulary)
        self.tgt2src = self.init_mapping(tgt_to_src_dict_filename, self.tgt_vocabulary, self.src_vocabulary)

    @staticmethod
    def init_mapping(bi_dict_filename: str, first_vocab: Vocabulary, second_vocab: Vocabulary):
        mapping = {0: 0, 1: 1, 2: 2, 3: 3}
        with open(bi_dict_filename, "r", encoding='utf-8') as r:
            for line in r:
                first_word, second_word = line.strip().split()
                first_index = first_vocab.get_index(first_word)
                second_index = second_vocab.get_index(second_word)
                mapping[first_index] = second_index
        return mapping

    def translate_src2tgt(self, variable: Variable, lengths):
        return self.map_variable(variable, self.src2tgt)

    def translate_tgt2src(self, variable: Variable, lengths):
        return self.map_variable(variable, self.tgt2src)

    def map_variable(self, variable: Variable, mapping: Dict[int, int]):
        input_max_length = variable.size(0)
        batch_size = variable.size(1)

        # Mapping
        output_variable = Variable(torch.zeros(self.max_length, batch_size)).type(torch.LongTensor)
        for t in range(input_max_length):
            for i in range(batch_size):
                index = variable[t, i].data[0]
                if index in mapping:
                    output_variable[t, i] = mapping[index]
                elif index != 0:
                    output_variable[t, i] = self.src_vocabulary.get_unk()

        # Padding
        for i in range(batch_size):
            eos_index = self.max_length - 1
            for t in range(self.max_length):
                if output_variable[t, i].data[0] == 2:
                    eos_index = t
                    break
            for t in range(eos_index + 1, self.max_length):
                output_variable[t, i] = 0
        return output_variable

    def translate_src2tgt_sentence(self, sentence: str):
        indices = indices_from_sentence(sentence, self.src_vocabulary)
        variable = self.indices_to_variable(indices)
        output_variable = self.translate_src2tgt(variable, None)
        output_variable = output_variable.transpose(0, 1)
        tgt_indices = [i for i in list(output_variable[0].data) if i != 0]
        result = [self.tgt_vocabulary.get_word(i) for i in tgt_indices]
        return result

    def translate_tgt2src_sentence(self, sentence: str):
        indices = indices_from_sentence(sentence, self.tgt_vocabulary)
        variable = self.indices_to_variable(indices)
        output_variable = self.translate_tgt2src(variable, None)
        output_variable = output_variable.transpose(0, 1)
        src_indices = [i for i in list(output_variable[0].data) if i != 0]
        result = [self.src_vocabulary.get_word(i) for i in src_indices]
        return result

    def indices_to_variable(self, indices: List[int]):
        batch_size = 32
        variable = Variable(torch.zeros(self.max_length, batch_size)).type(torch.LongTensor)
        indices = indices[:self.max_length]
        for i, index in enumerate(indices):
            variable[i, 0] = index
        for b in range(1, batch_size):
            variable[0, b] = 2
        return variable
