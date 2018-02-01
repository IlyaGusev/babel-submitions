# import torch
# from src.train import Trainer
#
# use_cuda = torch.cuda.is_available()
# print("Use CUDA: ", use_cuda)
#
# SRC_LANG = "en"
# TGT_LANG = "ru"
# SRC_TO_TGT_DICT_FILENAME = "models/" + SRC_LANG + "-" + TGT_LANG + ".txt"
# TGT_TO_SRC_DICT_FILENAME = "models/" + TGT_LANG + "-" + SRC_LANG + ".txt"
# SRC_EMBEDDINGS = "models/wiki.multi." + SRC_LANG + ".vec"
# TGT_EMBEDDINGS = "models/wiki.multi." + TGT_LANG + ".vec"
# SRC_CORPUS = "data/corpus.tok.clean.tc." + SRC_LANG
# TGT_CORPUS = "data/corpus.tok.clean.tc." + TGT_LANG
#
# state = Trainer(SRC_LANG, TGT_LANG, use_cuda=use_cuda)
# state.init_model([SRC_CORPUS, ], [TGT_CORPUS, ], SRC_EMBEDDINGS, TGT_EMBEDDINGS, SRC_TO_TGT_DICT_FILENAME,
#                  TGT_TO_SRC_DICT_FILENAME, src_max_words=5000, tgt_max_words=5000, load_pretrained_embeddings=False,
#                  hidden_size=4)
# state.train_supervised([("data/parallel.tok.tc.en", "data/parallel.tok.tc.ru"), ],
#                        big_epochs=5, batch_size=2, n_batches=1000, print_every=2)
# state.train([SRC_CORPUS, ], [TGT_CORPUS, ], big_epochs=3, batch_size=8, print_every=2, n_batches=1000)

import argparse

import torch
from src.trainer import Trainer


def train_opts(parser):
    # Mode
    group = parser.add_argument_group('Mode')
    group.add_argument('-supervised_only', type=bool, default=False,
                       help='Flag for supervised only mode.')

    # Languages Options
    group = parser.add_argument_group('Languages')
    group.add_argument('-src_lang', type=str, required=True,
                       help='Src language.')
    group.add_argument('-tgt_lang', type=str, required=True,
                       help='Tgt language.')

    # Data options
    group = parser.add_argument_group('Data')
    group.add_argument('-train_src_mono', required=True,
                       help="Path to the training source monolingual data")
    group.add_argument('-train_tgt_mono', required=True,
                       help="Path to the training target monolingual data")
    group.add_argument('-train_src_bi', default=None,
                       help="Path to the training source bilingual data")
    group.add_argument('-train_tgt_bi', default=None,
                       help="Path to the training target bilingual data")
    group.add_argument('-n_batches', type=int, default=None,
                       help="Count of src batches to process")

    # Embedding Options
    group = parser.add_argument_group('Embeddings')
    group.add_argument('-src_embeddings', type=str, default=None,
                       help='Pretrained word embeddings for src language.')
    group.add_argument('-tgt_embeddings', type=str, default=None,
                       help='Pretrained word embeddings for tgt language.')

    # Zero Model Options
    group = parser.add_argument_group('Zero Model')
    group.add_argument('-src_to_tgt_dict', type=str, required=True,
                       help='Pretrained word embeddings for src language.')
    group.add_argument('-tgt_to_src_dict', type=str, required=True,
                       help='Pretrained word embeddings for tgt language.')

    # Encoder-Decoder Options
    group = parser.add_argument_group('Model-Encoder-Decoder')
    group.add_argument('-layers', type=int, default=3,
                       help='Number of layers in enc/dec.')
    group.add_argument('-rnn_size', type=int, default=300,
                       help='Size of rnn hidden states')

    # Dictionary options, for text corpus
    group = parser.add_argument_group('Vocab')
    group.add_argument('-src_vocab_size', type=int, default=50000,
                       help="Size of the source vocabulary")
    group.add_argument('-tgt_vocab_size', type=int, default=50000,
                       help="Size of the target vocabulary")

    # Model loading/saving options
    group = parser.add_argument_group('General')
    group.add_argument('-save_model', default='model',
                       help="""Model filename (the model will be saved as
                       <save_model>_epochN_PPL.pt where PPL is the
                       validation perplexity""")
    group.add_argument('-save_every', type=int, default=1000,
                       help='Count of minibatches to save')
    group.add_argument('-seed', type=int, default=1337,
                       help="""Random seed used for the experiments
                       reproducibility.""")

    # Init options
    # group = parser.add_argument_group('Initialization')
    # group.add_argument('-start_epoch', type=int, default=1,
    #                    help='The epoch from which to start')
    # group.add_argument('-train_from', default='', type=str,
    #                    help="""If training from a checkpoint then this is the
    #                    path to the pretrained model's state_dict.""")

    # Logging
    group = parser.add_argument_group('Logging')
    group.add_argument('-print_every', type=int, default=1000,
                       help='Count of minibatches to print')

    # Optimization options
    group = parser.add_argument_group('Optimization- Type')
    group.add_argument('-batch_size', type=int, default=64,
                       help='Maximum batch size for training')
    group.add_argument('-unsupervised_epochs', type=int, default=2,
                       help='Number of unsupervised training epochs')
    group.add_argument('-supervised_epochs', type=int, default=10,
                       help='Number of supervised training epochs')
    group.add_argument('-adam_beta1', type=float, default=0.5,
                       help="""The beta1 parameter used by Adam.
                       Almost without exception a value of 0.9 is used in
                       the literature, seemingly giving good results,
                       so we would discourage changing this value from
                       the default without due consideration.""")

    # learning rate
    group = parser.add_argument_group('Optimization- Rate')
    group.add_argument('-learning_rate', type=float, default=0.0003,
                       help="""Main learning rate.""")
    group.add_argument('-discr-learning_rate', type=float, default=0.0005,
                       help="""Discriminator learning rate""")


def translate_opts(parser):
    group = parser.add_argument_group('Model')
    group.add_argument('-model', required=True,
                       help='Path to model .pt file')

    group.add_argument('-src',  required=True,
                       help="""Source sequence to decode (one line per
                       sequence)""")
    group.add_argument('-output', default='pred.txt',
                       help="""Path to output the predictions (each line will
                       be the decoded sequence""")

    group = parser.add_argument_group('Efficiency')
    group.add_argument('-batch_size', type=int, default=30,
                       help='Batch size')

parser = argparse.ArgumentParser(
    description='train.py',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)

# opts.py
train_opts(parser)
opt = parser.parse_args()


def main():
    use_cuda = torch.cuda.is_available()
    print("Use CUDA: ", use_cuda)
    state = Trainer(opt.src_lang, opt.tgt_lang, use_cuda=use_cuda)
    state.init_model([opt.train_src_mono, ], [opt.train_tgt_mono, ],
                     src_to_tgt_dict_filename=opt.src_to_tgt_dict,
                     tgt_to_src_dict_filename=opt.tgt_to_src_dict,
                     src_embeddings_filename=opt.src_embeddings,
                     tgt_embeddings_filename=opt.tgt_embeddings,
                     src_max_words=opt.src_vocab_size,
                     tgt_max_words=opt.tgt_vocab_size,
                     hidden_size=opt.rnn_size,
                     n_layers=opt.layers,
                     discriminator_lr=opt.discr_learning_rate,
                     main_lr=opt.learning_rate,
                     main_betas=(opt.adam_beta1, 0.999))
    if not opt.supervised_only:
        state.train([opt.train_src_mono, ], [opt.train_tgt_mono, ],
                    big_epochs=opt.unsupervised_epochs,
                    batch_size=opt.batch_size,
                    print_every=opt.print_every,
                    save_every=opt.save_every,
                    save_file=opt.save_model,
                    n_batches=opt.n_batches)

    assert opt.train_src_bi is not None
    assert opt.train_tgt_bi is not None
    state.train_supervised([(opt.train_src_bi, opt.train_tgt_bi), ],
                           big_epochs=opt.supervised_epochs,
                           batch_size=opt.batch_size,
                           print_every=opt.print_every,
                           save_every=opt.save_every,
                           save_file=opt.save_model)

if __name__ == "__main__":
    main()
