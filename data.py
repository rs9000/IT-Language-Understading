import pandas as pd
from torch.utils.data import Dataset


class UD_Dataset(Dataset):

    def __init__(self, filename, mode):
        self.line = []
        self._filename = filename
        col_names = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
        self.df = pd.read_csv(self._filename, sep='\t', header=None, names=col_names)
        self._file_len = 13120 if mode == "train" else 480
        self.index = 7

    def __len__(self):
        return self._file_len

    def __getitem__(self, idx):
        try:
            sentence = pd.DataFrame()
            while self.df.values[self.index][0] != str(1):
                self.index += 1

            sentence = sentence.append(self.df.ix[self.index])
            row = 2
            self.index += 1

            while self.df.values[self.index][0] != str(1):
                if str(row) == self.df.values[self.index][0]:
                    sentence = sentence.append(self.df.ix[self.index])
                    row += 1
                self.index += 1
            return sentence
        except IndexError:
            self.index = 0
            return self.__getitem__(0)


'''
Dataset: Contiene il corpora italiano.

Il file viene letto una frase alla volta perchè
troppo costoso da caricare in Memoria.

__len__ : Restituisce la lunghezza in righe del file.
__getitem__: Restituisce una frase letta dal file e l'indice di riga
             dell'ultima lettura.
'''


class LazyTextDataset(Dataset):

    def __init__(self, filename):
        self.line = []
        self._filename = filename
        self._file_len = 277000000
        self.first_row = pd.DataFrame()
        col_names = ["0", "1", "2", "3", "4", "5", "6", "7"]
        self.reader = pd.read_csv(self._filename, sep='\t', header=None, skiprows=12, names=col_names, iterator=True)

    def __len__(self):
        return self._file_len

    def __getitem__(self, idx):

        sentence = self.first_row.append(self.reader.get_chunk(1), ignore_index=True)
        while sentence.values.shape[0] <= 1 or str(sentence.values[-1][0]) != str(1):
            sentence = sentence.append(self.reader.get_chunk(1))

        self.first_row = sentence.tail(1)
        sentence.drop(sentence.tail(1).index, inplace=True)
        return sentence
