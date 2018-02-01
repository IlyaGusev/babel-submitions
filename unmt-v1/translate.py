import argparse

import torch
from src.trainer import Trainer


def translate_opts(parser):
    # Languages Options
    group = parser.add_argument_group('Languages')
    group.add_argument('-src_lang', type=str, required=True,
                       help='Src language.')
    group.add_argument('-tgt_lang', type=str, required=True,
                       help='Tgt language.')

    group = parser.add_argument_group('Vocabulary')
    group.add_argument('-train_src_mono', required=True,
                       help="Path to the training source monolingual data")
    group.add_argument('-train_tgt_mono', required=True,
                       help="Path to the training target monolingual data")
    group.add_argument('-src_vocab_size', type=int, default=50000,
                       help="Size of the source vocabulary")
    group.add_argument('-tgt_vocab_size', type=int, default=50000,
                       help="Size of the target vocabulary")

    # Embedding Options
    group = parser.add_argument_group('Embeddings')
    group.add_argument('-src_embeddings', type=str, default=None,
                       help='Pretrained word embeddings for src language.')
    group.add_argument('-tgt_embeddings', type=str, default=None,
                       help='Pretrained word embeddings for tgt language.')

    group = parser.add_argument_group('Model')
    group.add_argument('-lang', type=str, default="src",
                       help='Src language (src/tgt)')
    group.add_argument('-layers', type=int, default=3,
                       help='Number of layers in enc/dec.')
    group.add_argument('-rnn_size', type=int, default=300,
                       help='Size of rnn hidden states')
    group.add_argument('-discriminator_hidden_size', type=int, default=512,
                       help='Size of discriminator hidden states')
    group.add_argument('-model', required=True,
                       help='Path to model .pt file')
    group.add_argument('-input',  required=True,
                       help="""Source sequence to decode (one line per
                       sequence)""")
    group.add_argument('-output', default='pred.txt',
                       help="""Path to output the predictions (each line will
                       be the decoded sequence""")

parser = argparse.ArgumentParser(
    description='translate.py',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)

# opts.py
translate_opts(parser)
opt = parser.parse_args()


def main():
    use_cuda = torch.cuda.is_available()
    print("Use CUDA: ", use_cuda)
    state = Trainer(opt.src_lang, opt.tgt_lang, use_cuda=use_cuda)
    state.init_model([opt.train_src_mono, ], [opt.train_tgt_mono, ],
                     src_max_words=opt.src_vocab_size,
                     tgt_max_words=opt.tgt_vocab_size,
                     src_embeddings_filename=opt.src_embeddings,
                     tgt_embeddings_filename=opt.tgt_embeddings,
                     hidden_size=opt.rnn_size,
                     n_layers=opt.layers,
                     discriminator_hidden_size=opt.discriminator_hidden_size)
    state.load(opt.model)
    input_filename = opt.input
    output_filename = opt.output
    lang = opt.lang
    with open(input_filename, "r", encoding="utf-8") as r, open(output_filename, "w", encoding="utf-8") as w:
        for line in r:
            translated = state.translate(line, lang)
            print(translated)
            w.write(translated+"\n")

if __name__ == "__main__":
    main()
