Use the Makefile to download and preprocess the data, then vectorize.py to
prepare the embeddings, vocabulary indices, etc.

e.g.

```
make subj.txt.conllu
make subj.target
make subj.split

python vectorize.py subj
```

