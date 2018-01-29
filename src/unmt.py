import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from src.models import EncoderRNN, AttnDecoderRNN, Generator


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
        self.out = nn.Linear(hidden_size, 2)

    def forward(self, encoder_output):
        max_length = encoder_output.size(0)
        batch_size = encoder_output.size(1)
        output = encoder_output.transpose(0, 1).contiguous().view(batch_size, max_length * self.encoder_hidden_size)
        output = F.pad(output, (0, (self.max_length - max_length) * self.encoder_hidden_size), "constant", 0)
        # S = batch_size, max_length * encoder_hidden_size
        for i in range(self.n_layers):
            output = self.layers[i](output)
            output = self.activation(output)
        return F.log_softmax(self.out(output), dim=1)


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
            if word in src_embeddings.wv and i > 3:
                aligned_src_embeddings[i] = torch.FloatTensor(src_embeddings.wv[word])

        aligned_tgt_embeddings = torch.div(torch.randn(self.tgt_vocabulary.size(), 300), 10)
        for i, word in enumerate(self.tgt_vocabulary.index2word):
            if word in tgt_embeddings.wv and i > 3:
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

    def forward(self, src_batch, tgt_batch, src_noisy_batch, tgt_noisy_batch,
                src_translated_noisy_batch, tgt_translated_noisy_batch,
                batch_size, src_criterion, tgt_criterion, src_vocabulary, tgt_vocabulary):
        src_adv_loss, src_auto_loss = \
            self.auto_encoder_decoder_run(self.src_encoder, self.src_decoder, self.src_generator, src_criterion,
                                          src_noisy_batch.variable, src_noisy_batch.lengths, batch_size, lang="src")

        tgt_adv_loss, tgt_auto_loss = \
            self.auto_encoder_decoder_run(self.tgt_encoder, self.tgt_decoder, self.tgt_generator, tgt_criterion,
                                          tgt_noisy_batch.variable, tgt_noisy_batch.lengths, batch_size, lang="tgt")

        cd_tgt_adv_loss, cd_tgt_cd_loss = \
            self.cd_encoder_decoder_run(self.src_encoder, self.tgt_decoder, self.tgt_generator, tgt_criterion,
                                        tgt_translated_noisy_batch.variable, tgt_translated_noisy_batch.lengths,
                                        tgt_batch.variable, batch_size, lang="tgt")

        cd_src_adv_loss, cd_src_cd_loss = \
            self.cd_encoder_decoder_run(self.tgt_encoder, self.src_decoder, self.src_generator, src_criterion,
                                        src_translated_noisy_batch.variable, src_translated_noisy_batch.lengths,
                                        src_batch.variable, batch_size, lang="src")

        print("Losses:", [src_adv_loss.data[0], tgt_adv_loss.data[0], cd_tgt_adv_loss.data[0], cd_src_adv_loss.data[0],
                          src_auto_loss.data[0], tgt_auto_loss.data[0], cd_tgt_cd_loss.data[0], cd_src_cd_loss.data[0]])
        return sum([src_adv_loss, src_auto_loss, tgt_adv_loss, tgt_auto_loss,
                    cd_tgt_adv_loss, cd_tgt_cd_loss, cd_src_adv_loss, cd_src_cd_loss])

    def translate_src2tgt(self, variable, lengths):
        return self.translate(variable, self.src_encoder, self.tgt_decoder, self.tgt_generator, lengths)

    def translate_tgt2src(self, variable, lengths):
        return self.translate(variable, self.tgt_encoder, self.src_decoder, self.src_generator, lengths)

    def translate(self, variable, encoder, decoder, generator, lengths):
        batch_size = variable.size(1)
        output_variable = Variable(torch.zeros(self.max_length, batch_size)).type(torch.LongTensor)
        output_variable = output_variable.cuda() if self.use_cuda else output_variable

        encoder_output, encoder_hidden = encoder(variable, lengths, None)
        initial_input, initial_context = decoder.init_state(batch_size)

        hidden = encoder_hidden
        current_input = initial_input
        current_context = initial_context
        for t in range(self.max_length):
            current_context, hidden = decoder(None, None, hidden, encoder_output,
                                              current_input, current_context, one_step=True)
            scores = generator(current_context.squeeze(0))
            indices = scores.topk(1, dim=1)[1]
            output_variable[t] = indices
        output_variable = output_variable.detach()
        return output_variable

    def auto_encoder_decoder_run(self, encoder, decoder, generator, criterion, variable,
                                 lengths, batch_size, lang="src"):

        encoder_output, encoder_hidden = encoder(variable, lengths, None)

        # Adversarial part
        adv_criterion = nn.NLLLoss()
        log_proba = self.discriminator(encoder_output)
        if lang == "src":
            target_variable = Variable(torch.LongTensor([1 for _ in range(batch_size)]))
        else:
            target_variable = Variable(torch.LongTensor([0 for _ in range(batch_size)]))
        target_variable = target_variable.cuda() if self.use_cuda else target_variable
        adv_loss = adv_criterion(log_proba, target_variable)

        # Auto part
        initial_input, initial_context = decoder.init_state(batch_size)
        decoder_output, _, _ = decoder(variable, lengths, encoder_hidden, encoder_output,
                                       initial_input, initial_context)
        auto_loss = 0
        max_length = max(lengths)
        for t in range(max_length):
            scores = generator(decoder_output[t])
            auto_loss += criterion(scores, variable[t])

        return adv_loss, auto_loss

    def cd_encoder_decoder_run(self, encoder, decoder, generator, criterion, variable, lengths,
                               gt_variable, batch_size, lang="src"):
        encoder_output, encoder_hidden = encoder(variable, lengths, None)

        # Adversarial part
        adv_criterion = nn.NLLLoss()
        log_proba = self.discriminator(encoder_output)
        if lang == "src":
            target_variable = Variable(torch.LongTensor([0 for _ in range(batch_size)]))
        else:
            target_variable = Variable(torch.LongTensor([1 for _ in range(batch_size)]))
        target_variable = target_variable.cuda() if self.use_cuda else target_variable
        adv_loss = adv_criterion(log_proba, target_variable)

        # Cross-domain part
        initial_input, initial_context = decoder.init_state(batch_size)
        decoder_output, _, _ = decoder(variable, lengths, encoder_hidden, encoder_output,
                                       initial_input, initial_context)

        cd_loss = 0
        max_length = gt_variable.size(0)
        for t in range(max_length):
            scores = generator(decoder_output[t])
            cd_loss += criterion(scores, gt_variable[t])

        return adv_loss, cd_loss