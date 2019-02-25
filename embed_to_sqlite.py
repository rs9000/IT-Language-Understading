import argparse
import sqlite3
import numpy
from gensim.models import FastText
import io


def parse_arguments():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--word_embed', type=str, default="../fasttext.bin",
                        help='Word embed (bin)', metavar='')
    parser.add_argument('--output', type=str, default="./fasttext.db",
                        help='Output sqlite', metavar='')
    return parser.parse_args()


def loadEmbedding(filename):
    print("Loading Word Embedding...")
    it_model = FastText.load_fasttext_format(filename, full_model=False)
    print("...Done!")
    return it_model


def adapt_array(array):
    """
    Using the numpy.save function to save a binary version of the array,
    and BytesIO to catch the stream of data and convert it into a sqlite3.Binary.
    """
    out = io.BytesIO()
    numpy.save(out, array)
    out.seek(0)

    return sqlite3.Binary(out.read())


def convert_array(blob):
    """
    Using BytesIO to convert the binary version of the array back into a numpy array.
    """
    out = io.BytesIO(blob)
    out.seek(0)

    return numpy.load(out)


def myiter(itmodel):
    for key in itmodel.wv.vocab:
        yield str(key), itmodel.wv[key]


def build_db(itmodel, output):
    # Register the new adapters
    sqlite3.register_adapter(numpy.ndarray, adapt_array)
    sqlite3.register_converter('array', convert_array)
    # Connect to a local database and create a table for the embeddings
    connection = sqlite3.connect(output, detect_types=sqlite3.PARSE_DECLTYPES)
    cursor = connection.cursor()
    cursor.execute('CREATE TABLE IF NOT EXISTS embeddings (key text, val array)')
    connection.commit()
    cursor.executemany('INSERT INTO embeddings (key, val) VALUES (?, ?)', myiter(itmodel))
    connection.commit()
    print("Sqlite db successful created! Path: " + str(output))


if __name__ == "__main__":
    args = parse_arguments()
    word_embed = loadEmbedding(args.word_embed)
    build_db(word_embed, args.output)