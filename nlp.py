import torch
from torch import nn
import re
import string
import sqlite3
import numpy
from embed_to_sqlite import adapt_array, convert_array
from retrying import retry


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
        sqlite3.register_adapter(numpy.ndarray, adapt_array)
        sqlite3.register_converter('array', convert_array)
        # Connect to a local database and create a table for the embeddings
        self.connection = sqlite3.connect(word_embed, detect_types=sqlite3.PARSE_DECLTYPES)

        # Bidirectional GRU
        self.bi_gru = torch.nn.GRU(input_size=350, hidden_size=100, num_layers=2, batch_first=True,
                                   dropout=0.5, bidirectional=True)

        self.char_gru = torch.nn.GRU(input_size=50, hidden_size=25, num_layers=2, batch_first=True,
                                     dropout=0.5, bidirectional=True).to(self.device)
        # MLP
        self.classifier = nn.Sequential(torch.nn.Linear(200, 200),
                                        nn.ReLU(inplace=True),
                                        torch.nn.Linear(200, self.out_size)
                                        )

        # Activation function: LogSoftmax
        self.probs = nn.LogSoftmax(dim=-1)

        # Word Embedding
        self.char_embed = nn.Embedding(len(self.char_vocab), 50, padding_idx=0).to(self.device)

        self.reset_parameters()

    def reset_parameters(self):
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
        text = re.sub('(\n|\r|\t)+', ' ', text)
        text = re.sub('ß', 'ss', text)
        text = re.sub('’', "'", text)
        text = re.sub('é', 'è', text)
        text = re.sub('[^a-zA-Z0-9.!?,;:\-\' àèìòù]+', '', text)
        text = re.sub(' +', ' ', text)
        return text

    @retry(wait_fixed=1000)
    def read_db(self, key):
        cursor = self.connection.cursor()
        cursor.execute('SELECT * FROM embeddings WHERE key=?', (key,))
        data = cursor.fetchone()
        return data[1] if data else None

    def word2tensor(self, word):
        word = self.read_db(word)
        if word is None:
            word = self.read_db("unk")
        w = torch.FloatTensor(word).to(self.device)
        return w

    def forward(self, df):

        word_sentence = []
        vect_word_sentence = []

        # Word-level embedding
        for i in range(len(df)):
            # Create list of words
            word = self.clean_text(str(df.values[i]))
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
        out = self.probs(self.classifier(out[:,:-1,:].squeeze(0)))
        return out
