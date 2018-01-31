import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from src.models import EncoderRNN, Generator, DecoderRNN, AttnDecoderRNN


class Discriminator(nn.Module):
    def __init__(self, max_length, encoder_hidden_size, hidden_size=1024, n_layers=3, activation=F.leaky_relu):
        super(Discriminator, self).__init__()

        self.encoder_hidden_size = encoder_hidden_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.activation = activation
        self.max_length = max_length

        layers = list()
        layers.append(nn.Linear(encoder_hidden_size * max_length, hidden_size))
        for i in range(n_layers - 1):
            layers.append(nn.Linear(hidden_size, hidden_size))
        self.layers = nn.ModuleList(layers)
        self.out = nn.Linear(hidden_size, 1)

    def forward(self, encoder_output):
        max_length = encoder_output.size(0)
        batch_size = encoder_output.size(1)
        output = encoder_output.transpose(0, 1).contiguous().view(batch_size, max_length * self.encoder_hidden_size)
        output = F.pad(output, (0, (self.max_length - max_length) * self.encoder_hidden_size), "constant", 0)
        # S = batch_size, max_length * encoder_hidden_size
        for i in range(self.n_layers):
            output = self.layers[i](output)
            output = self.activation(output)
        return F.sigmoid(self.out(output))


class UNMT(nn.Module):
    def __init__(self, embedding_dim, src_vocabulary, tgt_vocabulary, hidden_size,
                 encoder_n_layers=3, decoder_n_layers=3, dropout=0.1, max_length=50, use_cuda=True):
        super(UNMT, self).__init__()

        self.embedding_dim = embedding_dim
        self.src_size = src_vocabulary.size()
        self.tgt_size = tgt_vocabulary.size()
        self.hidden_size = hidden_size
        self.encoder_n_layers = encoder_n_layers
        self.decoder_n_layers = decoder_n_layers
        self.dropout = dropout
        self.max_length = max_length
        self.src_vocabulary = src_vocabulary
        self.tgt_vocabulary = tgt_vocabulary
        self.use_cuda = use_cuda

        self.src_encoder = EncoderRNN(self.src_size, embedding_dim, hidden_size, dropout=dropout,
                                      n_layers=encoder_n_layers)
        self.tgt_encoder = EncoderRNN(self.tgt_size, embedding_dim, hidden_size, dropout=dropout,
                                      n_layers=encoder_n_layers)
        self.src_decoder = AttnDecoderRNN(embedding_dim, hidden_size, self.src_size, dropout=dropout,
                                          max_length=max_length, n_layers=decoder_n_layers, use_cuda=use_cuda)
        self.tgt_decoder = AttnDecoderRNN(embedding_dim, hidden_size, self.tgt_size, dropout=dropout,
                                          max_length=max_length, n_layers=decoder_n_layers, use_cuda=use_cuda)
        self.src_generator = Generator(hidden_size, self.src_size)
        self.tgt_generator = Generator(hidden_size, self.tgt_size)
        self.discriminator = Discriminator(self.max_length, self.hidden_size)

    def load_embeddings(self, src_embeddings, tgt_embeddings, enable_training=False):
        aligned_src_embeddings = torch.div(torch.randn(self.src_vocabulary.size(), 300), 10)
        for i, word in enumerate(self.src_vocabulary.index2word):
            if word in src_embeddings.wv:
                aligned_src_embeddings[i] = torch.FloatTensor(src_embeddings.wv[word])

        aligned_tgt_embeddings = torch.div(torch.randn(self.tgt_vocabulary.size(), 300), 10)
        for i, word in enumerate(self.tgt_vocabulary.index2word):
            if word in tgt_embeddings.wv:
                aligned_tgt_embeddings[i] = torch.FloatTensor(tgt_embeddings.wv[word])

        self.src_encoder.embedding.weight = nn.Parameter(aligned_src_embeddings)
        self.tgt_encoder.embedding.weight = nn.Parameter(aligned_tgt_embeddings)
        self.src_decoder.embedding.weight = nn.Parameter(aligned_src_embeddings)
        self.tgt_decoder.embedding.weight = nn.Parameter(aligned_tgt_embeddings)

        if not enable_training:
            self.src_encoder.embedding.weight.requires_grad = False
            self.tgt_encoder.embedding.weight.requires_grad = False
            self.src_decoder.embedding.weight.requires_grad = False
            self.tgt_decoder.embedding.weight.requires_grad = False

    def forward(self, src_batch, tgt_batch, src_noisy_batch, tgt_noisy_batch, src_batch_, tgt_batch_,
                src_translated_noisy_batch, tgt_translated_noisy_batch, src_batch__, tgt_batch__,
                batch_size, src_criterion, tgt_criterion, src_vocabulary, tgt_vocabulary):
        adv_ones_variable = Variable(torch.add(torch.ones((batch_size,)), -0.1), requires_grad=False)
        adv_ones_variable = adv_ones_variable.cuda() if self.use_cuda else adv_ones_variable
        adv_zeros_variable = Variable(torch.add(torch.zeros((batch_size,)), 0.1), requires_grad=False)
        adv_zeros_variable = adv_zeros_variable.cuda() if self.use_cuda else adv_zeros_variable

        src_adv_loss, src_auto_loss = \
            self.encoder_decoder_run(self.src_encoder, self.src_decoder, self.src_generator, src_criterion,
                                     src_noisy_batch.variable, src_noisy_batch.lengths,
                                     src_batch_.variable, src_batch_.lengths, batch_size, adv_ones_variable)

        tgt_adv_loss, tgt_auto_loss = \
            self.encoder_decoder_run(self.tgt_encoder, self.tgt_decoder, self.tgt_generator, tgt_criterion,
                                     tgt_noisy_batch.variable, tgt_noisy_batch.lengths,
                                     tgt_batch_.variable, tgt_batch_.lengths, batch_size, adv_zeros_variable)
        return sum([src_adv_loss, src_auto_loss, tgt_adv_loss, tgt_auto_loss])

#         cd_src_adv_loss, cd_src_cd_loss = \
#             self.encoder_decoder_run(self.tgt_encoder, self.src_decoder, self.src_generator, src_criterion,
#                                      src_translated_noisy_batch.variable, src_translated_noisy_batch.lengths,
#                                      src_batch__.variable, src_batch__.lengths, batch_size, adv_zeros_variable)

#         cd_tgt_adv_loss, cd_tgt_cd_loss = \
#             self.encoder_decoder_run(self.src_encoder, self.tgt_decoder, self.tgt_generator, tgt_criterion,
#                                      tgt_translated_noisy_batch.variable, tgt_translated_noisy_batch.lengths,
#                                      tgt_batch__.variable, tgt_batch__.lengths, batch_size, adv_ones_variable)

#         return sum([src_adv_loss, src_auto_loss, tgt_adv_loss, tgt_auto_loss,
#                     cd_tgt_adv_loss, cd_tgt_cd_loss, cd_src_adv_loss, cd_src_cd_loss])

    def encoder_decoder_run(self, encoder, decoder, generator, criterion, variable, lengths,
                            gt_variable, gt_lengths, batch_size, adv_variable):
        encoder_output, encoder_hidden = encoder(variable, lengths, None)

        adv_loss = self.get_discriminator_loss(encoder_output, adv_variable)

        main_loss = 0
#         initial_input = decoder.init_state(batch_size)
        initial_input, initial_context = decoder.init_state(batch_size)
#         decoder_output, _ = decoder(gt_variable, gt_lengths, encoder_hidden, initial_input)
        decoder_output, _, _ = decoder(gt_variable, gt_lengths, encoder_hidden, encoder_output, initial_input, initial_context)
        max_length = max(gt_lengths)
        for t in range(max_length):
            scores = generator(decoder_output[t])
            main_loss += criterion(scores, gt_variable[t])
        return adv_loss, main_loss

    def get_discriminator_loss(self, encoder_output, target_variable):
        adv_criterion = nn.BCELoss()
        log_prob = self.discriminator(encoder_output)
        log_prob = log_prob.view(-1)
        adv_loss = adv_criterion(log_prob, target_variable)
        return adv_loss

    def translate(self, variable, encoder, decoder, generator, lengths):
        self.eval()
        batch_size = variable.size(1)
        output_variable = Variable(torch.zeros(self.max_length, batch_size)).type(torch.LongTensor)
        output_variable = output_variable.cuda() if self.use_cuda else output_variable

        encoder_output, hidden = encoder(variable, lengths, None)
#         current_input = decoder.init_state(batch_size)
        current_input, context = decoder.init_state(batch_size)
        for t in range(self.max_length):
#             output, hidden = decoder(None, None, hidden, current_input, one_step=True)
            output, hidden, _ = decoder(None, None, hidden, encoder_output, current_input, context, one_step=True)
            context = output
        
            scores = generator(output.squeeze(0))
            top_indices = scores.topk(1, dim=1)[1].view(-1)
            output_variable[t] = top_indices
            current_input = top_indices
        output_variable = output_variable.detach()
        return output_variable

    def translate_src2tgt(self, variable, lengths):
        return self.translate(variable, self.src_encoder, self.tgt_decoder, self.tgt_generator, lengths)

    def translate_src_auto(self, variable, lengths):
        return self.translate(variable, self.src_encoder, self.src_decoder, self.src_generator, lengths)

    def translate_tgt2src(self, variable, lengths):
        return self.translate(variable, self.tgt_encoder, self.src_decoder, self.src_generator, lengths)
    
    def translate_tgt_auto(self, variable, lengths):
        return self.translate(variable, self.tgt_encoder, self.tgt_decoder, self.tgt_generator, lengths)
