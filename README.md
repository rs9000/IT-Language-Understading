# IT-Language-Understading
Deep Learning toolkit to Italian Natural Language Understanding
<br><br>
(Work in progress)

#### Word embedding
This model require Fasttext or another word embedding <br>
https://fasttext.cc/docs/en/crawl-vectors.html

#### Word embedding to Sqlite:
Sqlite provide fast access to word vectors reducing main memory usage
```
usage: embed_to_sqlite.py [-args]

arguments:
  --word_embed        Word embedding file (bin)
  --output            Output sqlite database
```
