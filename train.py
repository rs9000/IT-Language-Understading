from nlp import NLP
from tensorboardX import SummaryWriter
import pandas as pd
import torch
import argparse
from torch.utils.data import Dataset
import json
import nltk
from nltk.tokenize import word_tokenize


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
    parser.add_argument('--validation', type=bool, default=False,
                        help='Evaluate model', metavar='')
    parser.add_argument('--eval', type=str, default="",
                        help='Text sentence', metavar='')
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


'''
Val: Tagga frase

Args:
   sentence: frase testuale
   
'''


def val(sentence):
    try:
        sentence = word_tokenize(sentence)
    except:
        nltk.download('punkt')
        val(sentence)
        return

    df = pd.Series(sentence)
    out = model(df)

    for pred_label, word in zip(out, sentence):
        _, pred_label = torch.max(pred_label, 0)
        pred_label = tags[pred_label]
        print(word + " : " + pred_label)

    return 0


'''
Validation: Valutazione delle performance del modello.

Args:
    model: Modello
    data: Dataset
    tags: Dizionario dei tag
'''


def validation():

    acc = 0
    count = 0

    print("Start validation...")
    for step in range(5000):
        sample = data[step]
        out = model(sample['1'])

        for pred_label, true_label in zip(out, sample.values[:, 7]):
            _, pred_label = torch.max(pred_label, 0)
            true_label = tags.index(str(true_label))

            if true_label == pred_label.item():
                acc += 1
            count += 1

        if count > 1000:
            acc = acc / count
            print("Accuracy: " + str(acc))
            return acc


'''
Train: Loop di allenamento della rete.

Args:
    model: Modello
    data: Dataset
    tags: Dizionario dei tag
'''


def train():

    print("Start training...")
    criterion = torch.nn.NLLLoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)

    epoch = 0
    best_val = 0
    max_step = 5000

    acc = validation()
    writer.add_scalar('Test Accuracy', acc, epoch)
    while epoch < 10000:
        print("Epoch: " + str(epoch))
        for step in range(max_step):
            sample = data[step]
            out = model(sample['1'])

            label = []
            for e in sample.values[:, 7]:
                if str(e) in tags:
                    label.append(tags.index(str(e)))
                else:
                    print("Found new tag: " + str(e))
                    label.append(tags.index(str("nan")))
            label = torch.LongTensor(label).to(device)

            loss = criterion(out, label)
            loss.backward()
            writer.add_scalar('Train Loss', loss.item(), step + (max_step*epoch))
            #print('Train Loss: ' + str(loss.item()))
            optimizer.step()
            optimizer.zero_grad()
            step += 1

        epoch += 1
        acc = validation()
        writer.add_scalar('Test Accuracy', acc, epoch+1)
        if acc > best_val:
            best_val = acc
            torch.save(model.state_dict(), args.save_model)


'''
Entry Point

Vars:
    tags: Tag letti dal file json
    data: Dataset letto dal file corpora
    modello: modello 

'''


def main():
    if args.eval != "":
        model.load_state_dict(torch.load("checkpoint.pt"))
        val(args.eval)
        return 0

    if args.validation:
        model.load_state_dict(torch.load("checkpoint.pt"))
        validation()
    else:
        train()

    return 0


if __name__ == "__main__":
    args = parse_arguments()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    json_data = open(args.json).read()
    tags = json.loads(json_data)['d_tags']
    data = LazyTextDataset(args.corpora)
    model = NLP(args.word_embed, args.word_embed_size, len(tags), device).to(device)
    writer = SummaryWriter()
    main()
