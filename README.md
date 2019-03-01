# IT-Language-Understading
Deep Learning toolkit for Italian Natural Language Understanding
<br><br>
(Work in progress)

#### Data Source
https://github.com/UniversalDependencies/UD_Italian-ISDT

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

#### XPOS tagging

##### Test accuracy: 90.63%

```
usage: train.py [-args]

arguments:
  --load_model        ./checkpoint/xpos.pt
  --tag               xpos
  --eval              "Asimov scrisse... antologia."
```

##### Sentence: <br>
Asimov scrisse diversi racconti degni di nota, molti riguardanti i robot positronici e il Multivac racchiusi nell'antologia.

##### Output: <br>
Asimov : SP scrisse : V diversi : A racconti : S degni : B di : E nota : S , : FF molti : B riguardanti : V i : RD robot : S positronici : A e : CC il : RD Multivac : S racchiusi : A nell'antologia : SP . : FS
