# -*- coding: utf-8 -*-
"""
Description : Train an n-gram model.
Authors     : lihp
Time        : 2021/7/2
"""
from argparse import ArgumentParser
from n_gram import NGram

if __name__ == '__main__':
    parser = ArgumentParser(__file__)
    parser.add_argument('--corpus_path', '-cp', type=str, help='Path to the corpus for training the n-gram model.')
    parser.add_argument('--token_path', '-tp', type=str, help='Path to the token set of the n-gram model.')
    parser.add_argument('--model_path', '-mp', type=str, help='Path to save the n-gram model.')
    parser.add_argument('--n', '-n', type=int, help='The n of n-gram.')

    args = parser.parse_args()

    ng = NGram(args.n, args.token_path)
    ng.fix_v2(args.corpus_path)
    ng.save(args.model_path)
