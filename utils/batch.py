from utils.vocabulary import Vocabulary
from typing import List, Tuple
from utils.tqdm import tqdm_open
import torch
from torch.autograd import Variable


class OneLangBatch:
    def __init__(self, variable, lengths):
        self.variable = variable
        self.lengths = lengths

    def cuda(self):
        return OneLangBatch(self.variable.cuda(), self.lengths)

    def __str__(self):
        return "OneLangBatch: " + str(self.variable) + ", " + str(self.lengths)

    def __repr__(self):
        return self.__str__()


def indices_from_sentence(sentence: str, vocabulary: Vocabulary, lang):
    return [vocabulary.get_lang_index(word, lang) for word in sentence.split(' ')] + \
           [vocabulary.get_lang_eos(lang)]


def pad_seq(seq: List[int], vocabulary: Vocabulary, max_length: int):
    seq += [vocabulary.get_pad() for _ in range(max_length - len(seq))]
    return seq


class OneLangBatchGenerator:
    def __init__(self, filenames: List[str], batch_size: int, max_sentence_len: int, vocabulary: Vocabulary, lang: str):
        self.filenames = filenames  # type: List[str, str]
        self.batch_size = batch_size  # type: int
        self.max_sentence_len = max_sentence_len  # type: int
        self.vocabulary = vocabulary
        self.lang = lang

    def __iter__(self):
        for filename in self.filenames:
            seqs = []
            with tqdm_open(filename, encoding='utf-8') as r:
                for sentence in r:
                    sentence = sentence.strip()
                    sentence = indices_from_sentence(sentence, self.vocabulary, self.lang)
                    if len(sentence) >= self.max_sentence_len - 1 or len(sentence) >= self.max_sentence_len - 1:
                        continue

                    seqs.append(sentence)
                    if len(seqs) == self.batch_size:
                        yield self.__process(seqs)
                        seqs = []
            if len(seqs) == self.batch_size:
                yield self.__process(seqs)

    def __process(self, seqs):
        padded, lengths = self.__pad(seqs)
        variable = self.__to_tensor(padded)
        return OneLangBatch(variable, lengths)

    def __pad(self, seqs):
        seqs = sorted(seqs, key=lambda p: len(p), reverse=True)
        lengths = [len(s) for s in seqs]
        padded = [pad_seq(s, self.vocabulary, max(lengths)) for s in seqs]
        return padded, lengths

    def __to_tensor(self, padded):
        # Turn padded arrays into (batch_size x max_len) tensors, transpose into (max_len x batch_size)
        variable = Variable(torch.LongTensor(padded), requires_grad=False).transpose(0, 1)
        return variable
