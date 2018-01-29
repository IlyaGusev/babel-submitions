import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim

import time
import numpy as np

from utils.batch import OneLangBatch, BatchGenerator, OneLangBatchGenerator, indices_from_sentence
from src.word_by_word import WordByWordModel, inflate_vocabularies
from src.unmt import UNMT


class Trainer:
    def __init__(self, src_to_tgt_dict_filename: str, tgt_to_src_filename: str, src_lang: str, tgt_lang: str,
                 src_embeddings: str, tgt_embeddings: str, batch_size: int=64, max_length: int=50, use_cuda=True):
        self.batch_size = batch_size
        self.max_length = max_length
        self.use_cuda = use_cuda

        self.src_vocabulary, self.tgt_vocabulary = \
            inflate_vocabularies(src_to_tgt_dict_filename, tgt_to_src_filename, src_lang, tgt_lang)
        self.current_model = \
            WordByWordModel(src_to_tgt_dict_filename, tgt_to_src_filename, self.src_vocabulary, self.tgt_vocabulary)
        #         self.src_word_vectors = KeyedVectors.load_word2vec_format(src_embeddings, binary=False)
        #         self.tgt_word_vectors = KeyedVectors.load_word2vec_format(tgt_embeddings, binary=False)

        self.discriminator_optimizer = None
        self.main_optimizer = None

        weight = torch.ones(self.tgt_vocabulary.size())
        weight[self.tgt_vocabulary.get_pad()] = 0
        weight = weight.cuda() if self.use_cuda else weight
        self.tgt_criterion = nn.NLLLoss(weight, size_average=False)

        weight = torch.ones(self.src_vocabulary.size())
        weight[self.src_vocabulary.get_pad()] = 0
        weight = weight.cuda() if self.use_cuda else weight
        self.src_criterion = nn.NLLLoss(weight, size_average=False)

    def train(self, src_filenames, tgt_filenames, big_epochs: int, print_every=1000, save_every=1000, hidden_size=500):
        model = UNMT(300, self.src_vocabulary, self.tgt_vocabulary, hidden_size)
        #         model.load_embeddings(self.src_word_vectors, self.tgt_word_vectors, enable_training=False)
        model = model.cuda() if self.use_cuda else model

        self.discriminator_optimizer = optim.Adam(model.discriminator.parameters(), lr=0.0001)
        self.main_optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.0001)

        src_batches = self.get_one_lang_batches(src_filenames)
        tgt_batches = self.get_one_lang_batches(tgt_filenames)

        print(model)
        model_parameters = filter(lambda p: p.requires_grad, model.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        print("Params: ", params)

        for big_epoch in range(big_epochs):
            timer = time.time()
            print_loss_total = 0
            count_tokens = 0
            for epoch, (src_batch, tgt_batch) in enumerate(zip(src_batches, tgt_batches)):
                discriminator_loss, main_loss = self.train_batch(model, src_batch, tgt_batch)
                self.current_model = model
                print(discriminator_loss, main_loss)

                print_loss_total += main_loss
                count_tokens += sum(src_batch.lengths)
                if epoch % save_every == 0 and epoch != 0:
                    val_loss = 0
                    print_loss_avg = print_loss_total / print_every
                    print_loss_total = 0
                    diff = time.time() - timer
                    timer = time.time()
                    src_speed = count_tokens / diff
                    print('%s big epoch, %s, %s src tok/s, %s sec, %.4f loss, %.4f val loss' %
                          (big_epoch, epoch, src_speed, diff, print_loss_avg, val_loss))
                    count_tokens = 0

    def get_one_lang_batches(self, filenames, lang="src", n=1000):
        vocabulary = self.src_vocabulary if lang == "src" else self.tgt_vocabulary
        batch_generator = OneLangBatchGenerator(filenames, self.batch_size, self.max_length, vocabulary)
        batches = []
        i = 0
        for batch in batch_generator:
            batches.append(batch)
            if i == n:
                break
            i += 1
        return batches

    def get_parallel_batches(self, pair_filenames, n=1000):
        batch_generator = BatchGenerator(pair_filenames, self.batch_size, self.max_length,
                                         self.src_vocabulary, self.tgt_vocabulary, use_cuda)
        batches = []
        i = 0
        for batch in batch_generator:
            batches.append(batch)
            if i == n:
                break
            i += 1
        return batches

    def train_batch(self, model, src_batch: OneLangBatch, tgt_batch: OneLangBatch):
        # Disciminator step
        self.discriminator_optimizer.zero_grad()
        adv_criterion = nn.NLLLoss()

        src_batch = src_batch.cuda() if self.use_cuda else src_batch
        tgt_batch = tgt_batch.cuda() if self.use_cuda else tgt_batch

        src_encoder_output, _ = model.src_encoder(src_batch.variable, src_batch.lengths, None)
        log_proba = model.discriminator(src_encoder_output)
        src_variable = Variable(torch.LongTensor([0 for _ in range(self.batch_size)]))
        src_variable = src_variable.cuda() if self.use_cuda else src_variable
        src_adv_loss = adv_criterion(log_proba, src_variable)

        tgt_encoder_output, _ = model.tgt_encoder(tgt_batch.variable, tgt_batch.lengths, None)
        log_proba = model.discriminator(tgt_encoder_output)
        tgt_variable = Variable(torch.LongTensor([1 for _ in range(self.batch_size)]))
        tgt_variable = tgt_variable.cuda() if self.use_cuda else tgt_variable
        tgt_adv_loss = adv_criterion(log_proba, tgt_variable)

        discriminator_loss = src_adv_loss + tgt_adv_loss
        discriminator_loss.backward()
        nn.utils.clip_grad_norm(model.discriminator.parameters(), 5)
        self.discriminator_optimizer.step()

        # Main step
        src_noisy_batch = self.prepare_noisy_input(src_batch)
        tgt_noisy_batch = self.prepare_noisy_input(tgt_batch)

        src_translated_noisy_batch = self.prepare_translated_noisy_input(src_batch)
        tgt_translated_noisy_batch = self.prepare_translated_noisy_input(tgt_batch)

        src_noisy_batch = src_noisy_batch.cuda() if self.use_cuda else src_noisy_batch
        tgt_noisy_batch = tgt_noisy_batch.cuda() if self.use_cuda else tgt_noisy_batch
        src_translated_noisy_batch = src_translated_noisy_batch.cuda() if self.use_cuda else src_translated_noisy_batch
        tgt_translated_noisy_batch = tgt_translated_noisy_batch.cuda() if self.use_cuda else tgt_translated_noisy_batch

        self.main_optimizer.zero_grad()
        loss = model(src_batch, tgt_batch, src_noisy_batch, tgt_noisy_batch, src_translated_noisy_batch,
                     tgt_translated_noisy_batch,
                     self.batch_size, self.src_criterion, self.tgt_criterion, self.src_vocabulary, self.tgt_vocabulary)
        loss.backward()
        nn.utils.clip_grad_norm(model.parameters(), 5)
        self.main_optimizer.step()

        return discriminator_loss.data[0], loss.data[0]

    def get_variable(self, sentence, vocabulary):
        indices = indices_from_sentence(sentence, vocabulary)
        variable = Variable(torch.zeros(self.batch_size, len(indices))).type(torch.LongTensor)
        indices = Variable(torch.LongTensor(indices))
        variable[0] = indices
        for i in range(1, self.batch_size):
            variable[i, 0] = self.src_vocabulary.get_eos()
        variable = variable.transpose(0, 1)
        variable = variable.cuda() if self.use_cuda else variable
        lengths = [len(indices)]
        lengths += [1 for _ in range(self.batch_size - 1)]
        return variable, lengths

    def prepare_noisy_input(self, batch: OneLangBatch):
        new_variable, new_lengths = self.prepare_noisy_variable(batch.variable)
        return OneLangBatch(new_variable, new_lengths)

    def prepare_translated_noisy_input(self, batch: OneLangBatch, lang="src"):
        new_variable, _ = self.prepare_translated_variable(batch.variable, lang=lang)
        new_variable, new_lengths = self.prepare_noisy_variable(new_variable)
        return OneLangBatch(new_variable, new_lengths)

    def prepare_noisy_variable(self, variable):
        assert variable.size(1) == self.batch_size
        max_length = variable.size(0)
        variable = variable.transpose(0, 1)
        new_sentences = []
        for b in range(self.batch_size):
            indices = [elem for elem in variable[b].data if elem != 0][:-1]
            noisy = self.add_noise(indices) + [2, ]
            noisy = noisy + [0 for _ in range(max_length - len(noisy))]
            new_sentences.append(noisy)
        new_sentences = sorted(new_sentences, key=lambda p: len(p), reverse=True)

        new_varibale = Variable(torch.zeros(self.batch_size, max_length)).type(torch.LongTensor)
        new_lengths = []
        for sentence in new_sentences:
            new_varibale[b] = torch.LongTensor(sentence)
            new_lengths.append(len(sentence))
        return new_varibale.transpose(0, 1), new_lengths

    def prepare_translated_variable(self, variable, lang="src"):
        lengths = [len(variable[:, b].data) for b in range(variable.size(1))]
        new_sentences = []
        for b in range(self.batch_size):
            print("Input: ", list(variable[:, b].data))
            if lang == "src":
                translated = self.current_model.translate_src2tgt(variable, lengths)
            else:
                translated = self.current_model.translate_tgt2src(variable, lengths)
            translated = list(translated.transpose(0, 1)[b].data)
            print("Translated: ", translated)
            new_sentences.append(translated)
        new_sentences = sorted(new_sentences, key=lambda p: len(p), reverse=True)

        lengths = [len(sentence) for sentence in new_sentences]
        max_length = max(lengths)
        new_variable = Variable(torch.zeros(self.batch_size, max_length)).type(torch.LongTensor)
        for b in range(self.batch_size):
            current_sentence = new_sentences[b]
            current_sentence = current_sentence + [0 for _ in range(max_length - len(current_sentence))]
            new_variable[b] = torch.LongTensor(current_sentence)
        return new_variable.transpose(0, 1), lengths

    @staticmethod
    def add_noise(sequence, drop_probability=0.1, shuffle_max_distance=3):
        new_sequence = [elem for elem in sequence if np.random.random_sample() > drop_probability]
        new_sequence = [x for i, x in sorted(enumerate(new_sequence),
                                             key=lambda x: x[0] + (shuffle_max_distance + 1) * np.random.random())]
        return new_sequence