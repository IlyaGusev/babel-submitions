import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim

import time
import numpy as np
from gensim.models.keyedvectors import KeyedVectors

from utils.batch import OneLangBatch, OneLangBatchGenerator, indices_from_sentence
from src.word_by_word import WordByWordModel
from src.unmt import UNMT
from utils.vocabulary import Vocabulary
from utils.tqdm import tqdm_open
from src.models import EncoderRNN


class Trainer:
    def __init__(self, src_lang: str, tgt_lang: str, max_length: int=50, use_cuda=True):
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        self.max_length = max_length
        self.use_cuda = use_cuda

        self.src_vocabulary = None
        self.tgt_vocabulary = None
        self.src_criterion = None
        self.tgt_criterion = None

        self.model = None
        self.current_translation_model = None

        self.discriminator_optimizer = None
        self.main_optimizer = None

    def collect_vocabularies(self, src_filenames, tgt_filenames, src_max_words=80000, tgt_max_words=100000):
        print("Collecting vocabularies...")
        self.src_vocabulary = Vocabulary(language=self.src_lang)
        self.tgt_vocabulary = Vocabulary(language=self.tgt_lang)
        for filename in src_filenames:
            self.src_vocabulary = self.add_filename_to_vocabulary(filename, self.src_vocabulary)
        for filename in tgt_filenames:
            self.tgt_vocabulary = self.add_filename_to_vocabulary(filename, self.tgt_vocabulary)

        self.src_vocabulary.shrink(src_max_words)
        self.tgt_vocabulary.shrink(tgt_max_words)

        weight = torch.ones(self.tgt_vocabulary.size())
        weight[self.tgt_vocabulary.get_pad()] = 0
        weight = weight.cuda() if self.use_cuda else weight
        self.tgt_criterion = nn.NLLLoss(weight, size_average=False)

        weight = torch.ones(self.src_vocabulary.size())
        weight[self.src_vocabulary.get_pad()] = 0
        weight = weight.cuda() if self.use_cuda else weight
        self.src_criterion = nn.NLLLoss(weight, size_average=False)

    @staticmethod
    def add_filename_to_vocabulary(filename: str, vocabulary: Vocabulary):
        with tqdm_open(filename, encoding="utf-8") as r:
            for line in r:
                for word in line.strip().split():
                    vocabulary.add_word(word)
        return vocabulary

    def build_model(self, hidden_size, n_layers):
        print("Building model...")
        self.model = UNMT(300, self.src_vocabulary, self.tgt_vocabulary, hidden_size,
                          use_cuda=self.use_cuda, encoder_n_layers=n_layers, decoder_n_layers=n_layers)

    def load_embeddings(self, src_embeddings_filename, tgt_embeddings_filename, enable_training=False):
        print("Loading embeddings...")
        src_word_vectors = KeyedVectors.load_word2vec_format(src_embeddings_filename, binary=False)
        tgt_word_vectors = KeyedVectors.load_word2vec_format(tgt_embeddings_filename, binary=False)
        self.model.load_embeddings(src_word_vectors, tgt_word_vectors, enable_training=enable_training)

    def build_word_by_word_model(self, src_to_tgt_dict_filename, tgt_to_src_dict_filename):
        self.current_translation_model = WordByWordModel(src_to_tgt_dict_filename, tgt_to_src_dict_filename,
                                                         self.src_vocabulary, self.tgt_vocabulary)

    def init_model(self, src_filenames, tgt_filenames, src_embeddings_filename, tgt_embeddings_filename,
                   src_to_tgt_dict_filename, tgt_to_src_dict_filename, src_max_words=80000, tgt_max_words=100000,
                   load_pretrained_embeddings=True, hidden_size=200, n_layers=3, discriminator_lr=0.0005,
                   main_lr=0.0003, main_betas=(0.5, 0.999)):
        self.collect_vocabularies(src_filenames=src_filenames, tgt_filenames=tgt_filenames,
                                  src_max_words=src_max_words, tgt_max_words=tgt_max_words)
        self.build_model(hidden_size=hidden_size, n_layers=n_layers)
        if load_pretrained_embeddings:
            self.load_embeddings(src_embeddings_filename=src_embeddings_filename,
                                 tgt_embeddings_filename=tgt_embeddings_filename,
                                 enable_training=False)
        self.model = self.model.cuda() if self.use_cuda else self.model
        self.discriminator_optimizer = optim.RMSprop(self.model.discriminator.parameters(), lr=discriminator_lr)
        self.main_optimizer = optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()),
                                         lr=main_lr, betas=main_betas)

        self.build_word_by_word_model(src_to_tgt_dict_filename=src_to_tgt_dict_filename,
                                      tgt_to_src_dict_filename=tgt_to_src_dict_filename)
        print(self.model)
        model_parameters = filter(lambda p: p.requires_grad, self.model.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        print("Params: ", params)

    def train(self, src_filenames, tgt_filenames, big_epochs: int, print_every=1000, save_every=1000,
              batch_size: int = 32, n_batches=None):
        src_batches = self.get_one_lang_batches(src_filenames, lang="src", batch_size=batch_size, n=n_batches)
        tgt_batches = self.get_one_lang_batches(tgt_filenames, lang="tgt", batch_size=batch_size, n=n_batches)
        count_batches = min(len(src_batches), len(tgt_batches))

        for big_epoch in range(big_epochs):
            timer = time.time()
            print_main_loss_total = 0
            print_discriminator_loss_total = 0
            for epoch, (src_batch, tgt_batch) in enumerate(zip(src_batches, tgt_batches)):
                self.model.train()
                discriminator_loss, main_loss = self.train_batch(src_batch, tgt_batch)
                # print("Discriminator loss: ", discriminator_loss)
                # print("Main loss: ", main_loss)

                print_main_loss_total += main_loss
                print_discriminator_loss_total += discriminator_loss
                if epoch % save_every == 0 and epoch != 0:
                    self.save("model.pt")
                if epoch % print_every == 0 and epoch != 0:
                    print_main_loss_avg = print_main_loss_total / print_every
                    print_discriminator_loss_avg = print_discriminator_loss_total / print_every
                    print_main_loss_total = 0
                    print_discriminator_loss_total = 0
                    diff = time.time() - timer
                    timer = time.time()
                    print(self.autoencode("you can prepare your meals here .", lang="src"))
                    print(self.autoencode("по запросу могут приготовить другие блюда .", lang="tgt"))
                    print(self.translate("you can prepare your meals here .", lang="src"))
                    print('%s big epoch, %s/%s epoch, %s sec, %.4f main loss, %.4f discriminator loss' %
                          (big_epoch, epoch, count_batches, diff, print_main_loss_avg, print_discriminator_loss_avg))
            self.save("model-" + str(big_epoch) + ".pt")
            # self.current_translation_model = self.model

    def translate(self, sentence, lang):
        batch_size = 2
        vocabulary = self.src_vocabulary if lang == "src" else self.tgt_vocabulary
        translator = self.model.translate_src2tgt if lang == "src" else self.model.translate_tgt2src
        indices = indices_from_sentence(sentence, vocabulary)
        variable = Variable(torch.zeros(batch_size, len(indices))).type(torch.LongTensor)
        indices = Variable(torch.LongTensor(indices))
        variable[0] = indices
        for i in range(1, batch_size):
            variable[i, 0] = vocabulary.get_eos()
        variable = variable.transpose(0, 1)
        variable = variable.cuda() if self.use_cuda else variable
        lengths = [len(indices)]
        lengths += [1 for _ in range(batch_size - 1)]

        vocabulary = self.src_vocabulary if lang == "tgt" else self.tgt_vocabulary
        translated = list(translator(variable, lengths)[:, 0])
        return " ".join([vocabulary.get_word(i.data[0]) for i in translated
                         if i.data[0] != vocabulary.get_eos() and i.data[0] != vocabulary.get_pad()])

    def autoencode(self, sentence, lang):
        batch_size = 2
        vocabulary = self.src_vocabulary if lang == "src" else self.tgt_vocabulary
        translator = self.model.translate_src_auto if lang == "src" else self.model.translate_tgt_auto
        
        indices = indices_from_sentence(sentence, vocabulary)
        variable = Variable(torch.zeros(batch_size, len(indices))).type(torch.LongTensor)
        indices = Variable(torch.LongTensor(indices))
        variable[0] = indices
        for i in range(1, batch_size):
            variable[i, 0] = vocabulary.get_eos()
        variable = variable.transpose(0, 1)
        variable = variable.cuda() if self.use_cuda else variable
        lengths = [len(indices)]
        lengths += [1 for _ in range(batch_size - 1)]

        vocabulary = self.src_vocabulary if lang == "src" else self.tgt_vocabulary
        translated = list(translator(variable, lengths)[:, 0])
        return " ".join([vocabulary.get_word(i.data[0]) for i in translated
                         if i.data[0] != vocabulary.get_eos() and i.data[0] != vocabulary.get_pad()])

    @staticmethod
    def save_model(module, discriminator_optimizer, main_optimizer, filename):
        state_dict = module.state_dict()
        for key in state_dict.keys():
            state_dict[key] = state_dict[key].cpu()
        torch.save({
            'state_dict': state_dict,
            'discriminator_optimizer': discriminator_optimizer.state_dict(),
            'main_optimizer': main_optimizer.state_dict(),
        }, filename)

    def save(self, model_filename):
        Trainer.save_model(self.model, self.discriminator_optimizer, self.main_optimizer, model_filename)

    def load(self, model_filename):
        state_dict = torch.load(model_filename)
        self.model.load_state_dict(state_dict['state_dict'])
        self.discriminator_optimizer.load_state_dict(state_dict['discriminator_optimizer'])
        self.main_optimizer.load_state_dict(state_dict['main_optimizer'])

    def get_one_lang_batches(self, filenames, lang="src", batch_size: int=32, n=None):
        vocabulary = self.src_vocabulary if lang == "src" else self.tgt_vocabulary
        batch_generator = OneLangBatchGenerator(filenames, batch_size, self.max_length, vocabulary)
        batches = []
        i = 0
        for batch in batch_generator:
            batches.append(batch)
            if n is not None and i == n:
                break
            i += 1
        return batches

    def train_batch(self, src_batch: OneLangBatch, tgt_batch: OneLangBatch):
        batch_size = len(src_batch.lengths)
        src_batch = src_batch.cuda() if self.use_cuda else src_batch
        tgt_batch = tgt_batch.cuda() if self.use_cuda else tgt_batch

        discriminator_loss = self.discriminator_step(src_batch, tgt_batch)

        src_noisy_batch, src_batch_ = self.prepare_noisy_input(src_batch)
        tgt_noisy_batch, tgt_batch_ = self.prepare_noisy_input(tgt_batch)
        src_translated_noisy_batch,  src_batch__ = self.prepare_translated_noisy_input(src_batch, lang="src")
        tgt_translated_noisy_batch,  tgt_batch__ = self.prepare_translated_noisy_input(tgt_batch, lang="tgt")

        src_batch_ = src_batch_.cuda() if self.use_cuda else src_batch_
        tgt_batch_ = tgt_batch_.cuda() if self.use_cuda else tgt_batch_
        src_batch__ = src_batch__.cuda() if self.use_cuda else src_batch__
        tgt_batch__ = tgt_batch__.cuda() if self.use_cuda else tgt_batch__
        src_noisy_batch = src_noisy_batch.cuda() if self.use_cuda else src_noisy_batch
        tgt_noisy_batch = tgt_noisy_batch.cuda() if self.use_cuda else tgt_noisy_batch
        src_translated_noisy_batch = src_translated_noisy_batch.cuda() if self.use_cuda else src_translated_noisy_batch
        tgt_translated_noisy_batch = tgt_translated_noisy_batch.cuda() if self.use_cuda else tgt_translated_noisy_batch

        # Main step
        self.main_optimizer.zero_grad()
        loss = self.model(src_batch, tgt_batch, src_noisy_batch, tgt_noisy_batch, src_batch_, 
                          tgt_batch_, src_translated_noisy_batch, tgt_translated_noisy_batch, 
                          src_batch__, tgt_batch__, batch_size, self.src_criterion,
                          self.tgt_criterion, self.src_vocabulary, self.tgt_vocabulary)
        loss.backward()
        nn.utils.clip_grad_norm(self.model.parameters(), 5)
        self.main_optimizer.step()

        return discriminator_loss.data[0], loss.data[0]

    def discriminator_step(self, src_batch, tgt_batch):
        self.discriminator_optimizer.zero_grad()
        batch_size = len(src_batch.lengths)

        src_variable = Variable(torch.zeros((batch_size,)), requires_grad=False)
        src_variable = torch.add(src_variable, 0.1)
        src_variable = src_variable.cuda() if self.use_cuda else src_variable
        src_adv_loss = self.get_discriminator_loss(batch=src_batch, encoder=self.model.src_encoder,
                                                   target_variable=src_variable)

        tgt_variable = Variable(torch.ones((batch_size,)), requires_grad=False)
        tgt_variable = torch.add(tgt_variable, -0.1)
        tgt_variable = tgt_variable.cuda() if self.use_cuda else tgt_variable
        tgt_adv_loss = self.get_discriminator_loss(batch=tgt_batch, encoder=self.model.tgt_encoder,
                                                   target_variable=tgt_variable)
        discriminator_loss = src_adv_loss + tgt_adv_loss
        discriminator_loss.backward()
        nn.utils.clip_grad_norm(self.model.discriminator.parameters(), 5)
        self.discriminator_optimizer.step()

        return discriminator_loss

    def get_discriminator_loss(self, batch: OneLangBatch, encoder: EncoderRNN, target_variable: Variable):
        adv_criterion = nn.BCELoss()
        encoder_output, _ = encoder(batch.variable, batch.lengths, None)
        log_prob = self.model.discriminator(encoder_output).view(-1)
        return adv_criterion(log_prob, target_variable)

    def prepare_translated_noisy_input(self, batch: OneLangBatch, lang: str):
        translation = self.current_translation_model.translate_src2tgt if lang == "src" else \
            self.current_translation_model.translate_tgt2src

        new_variable, _ = self.prepare_translated_variable(translation, batch.variable)
        new_variable, new_lengths, permutation = self.prepare_noisy_variable(new_variable)

        new_old_batch = Trainer.get_permutated_batch(batch.variable, batch.lengths, permutation)
        return OneLangBatch(new_variable, new_lengths), new_old_batch

    @staticmethod
    def prepare_noisy_input(batch: OneLangBatch, drop_probability=0.1, shuffle_max_distance=3):
        new_variable, new_lengths, permutation = \
            Trainer.prepare_noisy_variable(batch.variable, drop_probability=drop_probability,
                                           shuffle_max_distance=shuffle_max_distance)
        new_old_batch = Trainer.get_permutated_batch(batch.variable, batch.lengths, permutation)
        return OneLangBatch(new_variable, new_lengths), new_old_batch

    @staticmethod
    def get_permutated_batch(variable, lengths, permutation):
        batch_size = variable.size(1)
        new_old_variable = Variable(torch.zeros(variable.size(0), variable.size(1)).type(torch.LongTensor))
        new_lengths = []
        for b in range(batch_size):
            new_old_variable[:, b] = variable[:, permutation[b]]
            new_lengths.append(lengths[permutation[b]])
        max_length = max(new_lengths)
        new_old_variable = new_old_variable[:max_length, :]
        return OneLangBatch(new_old_variable, new_lengths)

    @staticmethod
    def prepare_noisy_variable(variable: Variable, drop_probability=0.1, shuffle_max_distance=3):
        batch_size = variable.size(1)
        max_length = variable.size(0)
        variable = variable.transpose(0, 1)
        new_sentences = []
        for b in range(batch_size):
            noisy = Trainer.add_noise(variable[b].data.cpu().numpy(),
                                      drop_probability=drop_probability, shuffle_max_distance=shuffle_max_distance)
            noisy = noisy + [0] * (max_length - len(noisy))
            new_sentences.append((b, np.array(noisy)))
        new_sentences = sorted(new_sentences, key=lambda p: len([i for i in p[1] if i != 0]), reverse=True)
        permutation = [original_index for original_index, _ in new_sentences]
        new_sentences = [sentence for _, sentence in new_sentences]

        new_varibale = Variable(torch.zeros(batch_size, max_length)).type(torch.LongTensor)
        new_lengths = []
        for b, sentence in enumerate(new_sentences):
            new_varibale[b] = torch.LongTensor(sentence)
            new_lengths.append(len([i for i in sentence if i != 0]))
        return new_varibale.transpose(0, 1), new_lengths, permutation

    @staticmethod
    def prepare_translated_variable(translation, variable: Variable):
        def get_lengths(var):
            batch_size = var.size(1)
            sentences = [[index for index in var[:, b].data if index != 0] for b in range(batch_size)]
            return [len(sentence) for sentence in sentences]

        translated = translation(variable=variable, lengths=get_lengths(variable))
        translated.transpose(0, 1)

        lengths = get_lengths(translated)
        max_length = max(lengths)
        new_translated = translated[:max_length, :]

        return new_translated, lengths

    @staticmethod
    def add_noise(sequence, drop_probability=0.1, shuffle_max_distance=3):
        sequence = sequence[sequence > 0]
        sequence = sequence[:-1]
        sequence = sequence[np.random.random_sample(len(sequence)) > drop_probability]

        def perm(i):
            return i[0] + (shuffle_max_distance + 1) * np.random.random()
        sequence = [x for _, x in sorted(enumerate(sequence), key=perm)]
        sequence.append(2)
        return sequence
