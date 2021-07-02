# -*- coding: utf-8 -*-
"""
Description : Test the n-gram model.
Authors     : lihp
Time        : 2021/7/2
"""
from n_gram import NGram
from argparse import ArgumentParser


if __name__ == '__main__':
    parser = ArgumentParser(__file__)
    parser.add_argument('--token_path', '-tp', type=str, help='Path to the token set of the n-gram model.')
    parser.add_argument('--model_path', '-mp', type=str, help='Path to save the n-gram model.')
    parser.add_argument('--text', '-t', type=str, help='A text line for testing.')

    args = parser.parse_args()

    ng = NGram(args.token_path)
    ng.load(args.model_path)
    next_token = ng.predict_next_token(args.text)
    print(f"The most probable next token of the '{args.text}' is '{next_token}'.")

