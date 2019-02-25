import torch
from torch import nn
import re
import string
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import sqlite3
import numpy
from embed_to_sqlite import adapt_array, convert_array


def read_db(key):
    sqlite3.register_adapter(numpy.ndarray, adapt_array)
    sqlite3.register_converter('array', convert_array)
    # Connect to a local database and create a table for the embeddings
    connection = sqlite3.connect('./fasttext2.db', detect_types=sqlite3.PARSE_DECLTYPES)
    cursor = connection.cursor()
    cursor.execute('SELECT * FROM embeddings WHERE key=?', (key,))
    data = cursor.fetchone()
    return data[1] if data else None

'''
NPL:  Classe del modello

Metodi:
    reset_parameters: Inizializza layers della rete
    word2tensor: Word embedding
    forward: Sequence --> Tag
'''


class NLP(nn.Module):
    def __init__(self, word_embed, word_embed_size, out_size, device):
        super(NLP, self).__init__()

        self.size_embed = word_embed_size
        self.out_size = out_size
        self.device = device
        self.char_vocab = ['<pad>'] + list(string.printable) + ['à', 'è', 'ì', 'ò', 'ù', '<SOS>', '<EOS>']

        # Bidirectional GRU
        self.bi_gru = torch.nn.GRU(input_size=600, hidden_size=200, num_layers=2, batch_first=True,
                                   bidirectional=True)

        self.char_gru = torch.nn.GRU(input_size=100, hidden_size=150, num_layers=2, batch_first=True,
                                     bidirectional=True).to(self.device)
        # Linear Layer
        self.f1 = torch.nn.Linear(400, self.out_size)

        # Activation function: LogSoftmax
        self.probs = nn.LogSoftmax(dim=-1)

        # Word Embedding
        self.char_embed = nn.Embedding(len(self.char_vocab), 100, padding_idx=0).to(self.device)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.f1.weight, std=1)
        nn.init.normal_(self.f1.bias, std=0.01)

        for name, param in self.char_gru.named_parameters():
            if 'weight' in name:
                nn.init.xavier_normal_(param)
            if 'bias' in name:
                nn.init.constant_(param, 0)

        for name, param in self.bi_gru.named_parameters():
            if 'weight' in name:
                nn.init.xavier_normal_(param)
            if 'bias' in name:
                nn.init.constant_(param, 0)

    @staticmethod
    def clean_text(text):
        text = re.sub('<br />', '', text)
        text = re.sub('(\n|\r|\t)+', ' ', text)
        text = re.sub('ß', 'ss', text)
        text = re.sub('’', "'", text)
        text = re.sub('é', 'è', text)
        text = re.sub('[^a-zA-Z0-9.!?,;:\-\' äàâæçéèêîïíìöôóòœüûüúùÿ]+', '', text)
        text = re.sub(r'[^{0}\n]'.format(string.printable), '', text)
        text = re.sub(' +', ' ', text)
        return text

    def word2tensor(self, word):
        word = read_db(word)
        if word is None:
            word = read_db("unk")
        w = torch.FloatTensor(word).to(self.device)
        return w

    def forward(self, df):

        word_sentence = []
        vect_word_sentence = []

        # Word-level embedding
        for i in range(len(df)):
            # Create list of words
            word = self.clean_text(str(df.values[i][1]))
            if word == "":
                word = "unk"
            word_sentence.append(word)
            # Create list of embedded words
            w = self.word2tensor(word)
            vect_word_sentence.append(w)

        # Aggiunta end_token
        end_token = self.word2tensor("</s>")
        word_sentence.append("</s>")
        vect_word_sentence.append(end_token)

        vect_word_sentence = torch.stack(vect_word_sentence, 0).unsqueeze(0)

        # Character-level embedding
        char_sentence = [[self.char_vocab.index(char) for char in tok] for tok in word_sentence]
        vect_char_sentence = []

        for w in char_sentence:
            out = torch.LongTensor(w).to(self.device).unsqueeze(0)
            out = self.char_embed(out)
            out, _ = self.char_gru(out)
            vect_char_sentence.append(out[:,-1,:])
        vect_char_sentence = torch.stack(vect_char_sentence, 1)

        # JOIN Char and Word embeddings
        vect_word_sentence = torch.cat((vect_word_sentence, vect_char_sentence), -1)

        # (Word + Char) -> RNN
        out, h = self.bi_gru(vect_word_sentence)

        # L'output della RNN passa in una rete Feedforward
        # attivata da una funzione LogSoftmax
        out = self.probs(self.f1(out[:,:-1,:].squeeze(0)))
        return out
