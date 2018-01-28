class Vocabulary:
    def __init__(self, language):
        self.language = language
        self.word2index = {}
        self.word2count = Counter()
        self.index2word = ["<PAD>", "<SOS>", "<EOS>", "<UKN>"]
        if os.path.exists(self.language+".pickle"):
            self.load()

    def get_pad(self):
        return self.index2word.index("<PAD>")

    def get_sos(self):
        return self.index2word.index("<SOS>")

    def get_eos(self):
        return self.index2word.index("<EOS>")

    def get_ukn(self):
        return self.index2word.index("<UKN>")

    def add_sentence(self, sentence):
        for word in sentence.split(' '):
            if word == '':
                continue
            self.add_word(word)

    def add_word(self, word):
        if word not in self.word2index:
            self.word2index[word] = len(self.index2word)
            self.word2count[word] += 1
            self.index2word.append(word)
        else:
            self.word2count[word] += 1

    def get_index(self, word):
        if word in self.word2index:
            return self.word2index[word]
        else:
            return self.get_ukn()
        
    def get_word(self, index):
        return self.index2word[index]

    def size(self):
        return len(self.index2word)

    def is_empty(self):
        return self.size() <= 4

    def shrink(self, n):
        best_words = self.word2count.most_common(n)
        self.index2word = ["<PAD>", "<SOS>", "<EOS>", "<UKN>"]
        self.word2index = {}
        self.word2count = Counter()
        for word, count in best_words:
            self.add_word(word)
            self.word2count[word] = count

    def save(self) -> None:
        with open(self.language+".pickle", "wb") as f:
            pickle.dump(self, f, pickle.HIGHEST_PROTOCOL)

    def load(self):
        with open(self.language+".pickle", "rb") as f:
            vocab = pickle.load(f)
            self.__dict__.update(vocab.__dict__)