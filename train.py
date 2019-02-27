from nlp import NLP
from tensorboardX import SummaryWriter
import pandas as pd
import torch
import argparse
from data import UD_Dataset, LazyTextDataset
import json
import nltk
from nltk.tokenize import word_tokenize


def parse_arguments():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--train_data', type=str, default="./dataset/train.data",
                        help='Train dataset', metavar='')
    parser.add_argument('--test_data', type=str, default="./dataset/test.data",
                        help='Test dataset', metavar='')
    parser.add_argument('--json', type=str, default="./dataset/tags.json",
                        help='Tags annotations', metavar='')
    parser.add_argument('--tag', type=str, default="xpos",
                        help='Tags to predict', metavar='')
    parser.add_argument('--word_embed', type=str, default='./fasttext2.db',
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
    for step in range(int(len(data_val)/5)):
        sample = data_val[step]
        out = model(sample['1'])

        for pred_label, true_label in zip(out, sample.values[:, tag_idx]):
            _, pred_label = torch.max(pred_label, 0)
            true_label = tags.index(str(true_label))

            if true_label == pred_label.item():
                acc += 1
            count += 1

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
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    epoch = 0
    best_val = 0
    max_step = 500

    acc = validation()
    writer.add_scalar('Test Accuracy', acc, epoch)
    while epoch < 10000:
        print("Epoch: " + str(epoch))
        for step in range(max_step):
            sample = data_train[step]
            out = model(sample['1'])

            label = []
            for e in sample.values[:, tag_idx]:
                if str(e) in tags:
                    label.append(tags.index(str(e)))
                else:
                    print("Found new tag: " + str(e))
                    label.append(tags.index(str("nan")))
            label = torch.LongTensor(label).to(device)

            loss = criterion(out, label)
            loss.backward()
            writer.add_scalar('Train Loss', loss.item(), step + (max_step*epoch))
            torch.nn.utils.clip_grad_value_(model.parameters(), 10)
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
        model.load_state_dict(torch.load(args.save_model))
        val(args.eval)
        return 0

    if args.validation:
        model.load_state_dict(torch.load(args.save_model))
        validation()
    else:
        model.load_state_dict(torch.load(args.save_model))
        train()

    return 0


if __name__ == "__main__":
    args = parse_arguments()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    json_data = open(args.json).read()
    tags = json.loads(json_data)
    tag_list = ["id", "form", "lemma", "upos", "xpos", "feats", "head", "deprel", "deps", "misc"]

    tags = tags[args.tag]
    tag_idx = tag_list.index(args.tag)

    data_train, data_val = UD_Dataset(args.train_data, "train"), UD_Dataset(args.test_data, "test")
    model = NLP(args.word_embed, args.word_embed_size, len(tags), device).to(device)
    writer = SummaryWriter()
    main()
