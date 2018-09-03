import argparse
import numpy as np


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('file')
    parser.add_argument('--valid', type=float)
    parser.add_argument('--test', type=float, default=0)
    parser.add_argument('--seed', type=int, default=42)

    opts = parser.parse_args()

    if opts.test > 0:
        choices = ['train', 'valid', 'test']
        ps = [1 - (opts.valid + opts.test), opts.valid, opts.test]

    else:
        choices = ['train', 'valid']
        ps = [1 - opts.valid, opts.valid]

    choices = np.array(choices)
    ps = np.array(ps)

    rng = np.random.RandomState(opts.seed)

    with open(opts.file, encoding='latin-1') as f:
        size = len(f.readlines())

    out = rng.choice(choices, p=ps, size=size, replace=True)
    for val in out:
        print(val)

