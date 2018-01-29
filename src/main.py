import torch
from src.train import Trainer

use_cuda = torch.cuda.is_available()
print(use_cuda)

SRC_LANG = "en"
TGT_LANG = "ru"
SRC_TO_TGT_DICT_FILENAME = "models/" + SRC_LANG + "-" + TGT_LANG + ".txt"
TGT_TO_SRC_DICT_FILENAME = "models/" + TGT_LANG + "-" + SRC_LANG + ".txt"
SRC_EMBEDDINGS = "models/wiki.multi." + SRC_LANG + ".vec"
TGT_EMBEDDINGS = "models/wiki.multi." + TGT_LANG + ".vec"
SRC_CORPUS = "data/corpus1.txt"
TGT_CORPUS = "data/corpus2.txt"

state = Trainer(SRC_TO_TGT_DICT_FILENAME, TGT_TO_SRC_DICT_FILENAME, SRC_LANG, TGT_LANG,
                SRC_EMBEDDINGS, TGT_EMBEDDINGS, batch_size=2, use_cuda=use_cuda)
state.train([SRC_CORPUS, ], [TGT_CORPUS, ], 3, hidden_size=16)