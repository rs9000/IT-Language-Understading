from nlp import NLP
from tensorboardX import SummaryWriter
import pandas as pd
import torch
import argparse
from torch.utils.data import Dataset
import json


def parse_arguments():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--corpora', type=str, default="../corpora.utf8",
                        help='Corpora dataset', metavar='')
    parser.add_argument('--json', type=str, default="../tags.json",
                        help='Tags annotations', metavar='')
    parser.add_argument('--word_embed', type=str, default='../fasttext.bin',
                        help='Word embedding file', metavar='')
    parser.add_argument('--word_embed_size', type=int, default=300,
                        help='word embedding vector size', metavar='')
    parser.add_argument('--save_model', type=str, default='checkpoint.pt',
                        help='save model file', metavar='')
    parser.add_argument('--load_model', type=str, default='',
                        help='load model file', metavar='')
    parser.add_argument('--evaluation', type=bool, default=False,
                        help='Evaluate model', metavar='')
    return parser.parse_args()


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
        self._file_len = 278911616

    def __len__(self):
        return self._file_len

    def __getitem__(self, idx):
        col_names = ["0", "1", "2", "3", "4", "5", "6", "7"]
        df = pd.read_csv(self._filename, sep='\t', skiprows=idx, nrows=1000, header=None, names=col_names)
        for i in range(1, len(df)):
            if str(df.values[i][0]) == str(1):
                    df = df.head(i)
                    idx += i + 1
                    break
        return df, idx


'''
Validation: Valutazione delle performance del modello.

Args:
    model: Modello
    data: Dataset
    tags: Dizionario dei tag
'''


def validation(model, data, tags):

    idx = 12
    acc = 0
    count = 0

    while idx < len(data):
        sample, idx = data[idx]
        out = model(sample)

        for pred_label, true_label in zip(out, sample.values[:, 7]):
            _, pred_label = torch.max(pred_label, 0)
            true_label = tags.index(str(true_label))

            if true_label == pred_label.item():
                acc += 1
            count += 1

        if count > 50000:
            acc = acc / count
            print("Accuracy: " + str(acc))
            break


'''
Train: Loop di allenamento della rete.

Args:
    model: Modello
    data: Dataset
    tags: Dizionario dei tag
'''


def train(model, data, tags):

    print("Start training...")
    criterion = torch.nn.NLLLoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    epoch = 0
    idx = 12

    while idx < len(data):
        sample, idx = data[idx]
        out = model(sample)

        label = []
        for e in sample.values[:, 7]:
            label.append(tags.index(str(e)))
        label = torch.LongTensor(label).to(device)

        loss = criterion(out, label)

        loss.backward()
        print("Loss: " + str(loss.item()))
        optimizer.step()
        optimizer.zero_grad()

        torch.save(model.state_dict(), args.save_model)


'''
Entry Point

Vars:
    tags: Tag letti dal file json
    data: Dataset letto dal file corpora
    modello: modello 

'''
if __name__ == "__main__":
    args = parse_arguments()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    json_data = open(args.json).read()
    tags = json.loads(json_data)['d_tags']
    data = LazyTextDataset(args.corpora)
    model = NLP(args.word_embed, args.word_embed_size, len(tags), device).to(device)

    if not args.evaluation:
        train(model, data, tags)
    else:
        model.load_state_dict(torch.load("checkpoint.pt"))
        validation(model, data, tags)
