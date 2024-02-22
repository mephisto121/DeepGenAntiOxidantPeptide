import pickle

from sklearn.feature_extraction.text import CountVectorizer


class Vocab():
    def __init__(self, vocab):
        self.vocab = vocab
        self.inv_vocabulary = {v: k for k, v in vocab.items()}


    def save_data(self, x, name):
        data = open(f"{name}", "wb")
        pickle.dump(x, file=data)
        data.close()
    def __len__(self):
        return len(self.vocab)

    def int2str(self, num):
        return self.inv_vocabulary[num]


    def str2int(self, txt):
            return self.vocab[str(txt)]
    

    @classmethod
    def create_vocab(cls, seqs):
        count_vect = CountVectorizer(lowercase = False, analyzer = "char")
        cls.ready_seq = []
        for seq in seqs:
            cls.ready_seq.append(f"!{seq}%")
        count_vect.fit(cls.ready_seq)
        vocab_lst = count_vect.vocabulary_
        vocab_dict = {}
        for k,v in vocab_lst.items():
            vocab_dict[k] = v + 1
        vocab_dict['+'] = 0
        return Vocab(vocab_dict), cls.ready_seq
    
    @classmethod
    def creat_data(cls, seqs):
        cls.ready_seq = []
        for seq in seqs:
            cls.ready_seq.append(f"!{seq}%")
            
        return cls.ready_seq

