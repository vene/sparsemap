import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('file')
    opts = parser.parse_args()

    with open(opts.file) as f:
        sent, tree = [], []
        for line in f:
            line = line.strip()
            if not line:
                sent_s = " ".join(sent)
                tree_s = " ".join(tree)
                print(f"{sent_s}\t{tree_s}")

                sent = []
                tree = []
            else:
                fields = line.split("\t")
                word = fields[1]
                dep = fields[6]
                assert " " not in word
                assert "\t" not in word
                sent.append(word)
                tree.append(dep)

    if sent:
        sent_s = " ".join(sent)
        tree_s = " ".join(tree)
        print(f"{sent_s}\t{tree_s}")
