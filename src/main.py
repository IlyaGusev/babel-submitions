import torch
from src.train import Trainer

use_cuda = torch.cuda.is_available()
print("Use CUDA: ", use_cuda)

SRC_LANG = "en"
TGT_LANG = "ru"
SRC_TO_TGT_DICT_FILENAME = "models/" + SRC_LANG + "-" + TGT_LANG + ".txt"
TGT_TO_SRC_DICT_FILENAME = "models/" + TGT_LANG + "-" + SRC_LANG + ".txt"
SRC_EMBEDDINGS = "models/wiki.multi." + SRC_LANG + ".vec"
TGT_EMBEDDINGS = "models/wiki.multi." + TGT_LANG + ".vec"
SRC_CORPUS = "data/corpus.tok.clean.tc." + SRC_LANG
TGT_CORPUS = "data/corpus.tok.clean.tc." + TGT_LANG

state = Trainer(SRC_LANG, TGT_LANG, use_cuda=use_cuda)
state.init_model([SRC_CORPUS, ], [TGT_CORPUS, ], SRC_EMBEDDINGS, TGT_EMBEDDINGS, SRC_TO_TGT_DICT_FILENAME,
                 TGT_TO_SRC_DICT_FILENAME, src_max_words=5000, tgt_max_words=5000, load_pretrained_embeddings=True,
                 hidden_size=4)
state.train([SRC_CORPUS, ], [TGT_CORPUS, ], big_epochs=3, batch_size=8, print_every=2, n_batches=1000)
