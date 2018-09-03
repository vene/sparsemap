import argparse
from nltk import Tree

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('train')
    parser.add_argument('valid')
    parser.add_argument('test')
    opts = parser.parse_args()

    split = open("sst.split", "w")
    out = open("sst.txt", "w")
    tgt = open("sst.target", "w")

    for s, fn in zip(("train", "valid", "test"),
                     (opts.train, opts.valid, opts.test)):
        with open(fn) as f:
            for line in f:
                t = Tree.fromstring(line)
                label = t.label()
                sent = " ".join(t.leaves())
                sent = sent.replace("\\", "")

                print(sent, file=out)
                print(label, file=tgt)
                print(s, file=split)

    out.close()
    tgt.close()
    split.close()

