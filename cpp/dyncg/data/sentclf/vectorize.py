import argparse
import os
from collections import Counter

import numpy as np


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset')
    parser.add_argument('--glove',
        default="~/data/glove.840B.300d.txt")
    opts = parser.parse_args()

    with open(opts.dataset + ".target") as f:
        targets = [line.strip() for line in f]

    if opts.dataset == 'trec':
        targets = [tgt.split(":")[0] for tgt in targets]

    targets_uniq = sorted(list(set(targets)))
    targets_dict = {tgt: k for k, tgt in enumerate(targets_uniq)}

    with open(opts.dataset + ".split") as f:
        splits = [line.strip() for line in f]

    sents = []
    trees = []

    with open(opts.dataset + ".txt.indices") as f:
        for line in f:
            line = line.strip()
            sent, tree = line.split("\t")
            sents.append(sent)
            trees.append('-1 ' + tree)

    print(len(sents))
    print(len(trees))
    print(len(targets))
    print(len(splits))

    sents = [sent.strip().lower().split() for sent in sents]

    train_cnt = Counter(w for sent, split in zip(sents, splits)
                        if split == 'train'
                        for w in sent)

    print(train_cnt.most_common(10))

    vocab = [w for w, c in train_cnt.most_common() if c > 1]

    print(" ".join(vocab[:10]))
    print(" ".join(vocab[-10:]))

    vocab_set = set(vocab)

    embeds = dict()
    with open(os.path.expanduser(opts.glove)) as f:
        for line in f:
            word, emb = line.split(" ", 1)
            if word in vocab_set:
                embeds[word] = np.fromstring(emb, sep=' ', dtype=np.float32)

    embeds['__UNK__'] = sum(embeds.values()) / len(embeds)
    vocab = ['__UNK__'] + [w for w in vocab if w in embeds]
    bacov = {w: k for k, w in enumerate(vocab)}

    with open(opts.dataset + ".vocab", "w") as f:
        for w in vocab:
            print(w, file=f)

    E = np.row_stack([embeds[w] for w in vocab])
    np.savetxt(opts.dataset + ".embed", E)

    files = {spl: open("{}.{}.txt".format(opts.dataset, spl), "w")
             for spl in ('train', 'valid', 'test')}

    sorted_ix = range(len(sents))
    sorted_ix = sorted(sorted_ix, key=lambda k: len(sents[k]))

    for k in sorted_ix:
        sent = sents[k]
        target = targets[k]
        tree = trees[k]
        split = splits[k]

        sent_ix = [bacov.get(w.lower(), bacov['__UNK__']) for w in sent]
        sent_ix = " ".join(str(w) for w in sent_ix)
        line = "\t".join([str(targets_dict[target]), sent_ix, tree])
        print(line, file=files[split])

    for f in files.values():
        f.close()

    with open(opts.dataset + ".classes", "w") as f:
        for tgt in targets_uniq:
            print(tgt, file=f)
