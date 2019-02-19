import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import io
from gensim.models import FastText

'''
Embedding:  Funzione che carica il modello di word-embedding.
Computazionalmente molto costosa, richiede molta Memoria.
'''


def loadEmbedding(filename):
    print("Loading Word Embedding...")
    it_model = FastText.load_fasttext_format(filename, full_model=False)
    print("...Done!")
    return it_model


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

        # Bidirectional GRU
        self.bi_gru = torch.nn.GRU(input_size=self.size_embed, hidden_size=150, num_layers=1, batch_first=True,
                                   bidirectional=True)
        # Linear Layer
        self.f1 = torch.nn.Linear(300, self.out_size)

        # Activation function: LogSoftmax
        self.probs = nn.LogSoftmax(dim=-1)

        # Word Embedding
        self.words = loadEmbedding(word_embed)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.f1.weight, std=1)
        nn.init.normal_(self.f1.bias, std=0.01)

    def word2tensor(self, word):
        try:
            w = torch.tensor(self.words.wv[word]).to(self.device)
        except KeyError as error:
            w = torch.tensor(self.words.wv["<unk>"]).to(self.device)
        return w

    def forward(self, df):

        # Ogni token della frase viene trasformato in un vettore
        sentence = []
        for i in range(len(df)):
            word = str(df.values[i][1]).replace("'",'').replace('\t', '').replace('\n', '').lower()
            if word == '':
                word = '<unk>'
            w = self.word2tensor(word)
            sentence.append(w)

        # Aggiunta end_token
        end_token = self.word2tensor("</s>")
        sentence.append(end_token)

        # La sequenza di token viene eleborata da una RNN bidirezionale
        sentence = torch.stack(sentence, 0).unsqueeze(0)
        out, h = self.bi_gru(sentence)

        # L'output della RNN passa in una rete Feedforward
        # attivata da una funzione LogSoftmax
        out = self.probs(self.f1(out[:,:-1,:].squeeze(0)))
        return out
