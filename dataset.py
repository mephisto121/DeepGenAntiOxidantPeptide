import numpy as np
from sklearn.utils import shuffle
from tensorflow import keras
from keras.preprocessing.sequence import pad_sequences

from vocab import Vocab


class GenerateData(Vocab):
    def __init__(self, vocab):
        super().__init__(vocab)
        self.vocab = vocab

    def encode_data_lst(self, txt):
        en = []
        self.txt = txt
        for  seq in self.txt:
            m = []
            for i in seq:
                m.append(self.str2int(i))
            en.append(m)
        return en

    def encode_data_sngl(self, txt_sngl):
        en1 = []
        self.txt1 = txt_sngl
        for i in self.txt1:
            en1.append(self.str2int(i))
        return en1

    def decode_data(self, num_lst):
        decode_mol = []
        for i in num_lst:
            decode_mol.append(self.int2str(i))
        return "".join(decode_mol)

    def padding(self, encoded_data, mxlen, shuffle_ = True):
        if shuffle_ == True:
            self.encoded_data = shuffle(encoded_data, random_state=42)
            return pad_sequences(self.encoded_data, padding="post", maxlen=mxlen)
        elif shuffle_ == False:
            return pad_sequences(encoded_data, padding="post", maxlen=mxlen)  

    def split_input_target(self, sequence):
        input_text = sequence[:-1]
        target_text = sequence[1:]
        return input_text, target_text

    def split_for_data(self, encoded_data):
        self.data_lst = encoded_data
        self.input = []
        self.output = []

        for seq in self.data_lst:
            x, y = self.split_input_target(seq)
            self.input.append(np.asanyarray(x))
            self.output.append(np.asanyarray(y))
        return np.asanyarray(self.input), np.asanyarray(self.output)

    

