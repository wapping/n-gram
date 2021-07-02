# -*- coding: utf-8 -*-
"""
Description : N-gram language model.
Authors     : lihp
Time        : 2021/7/1
"""
import numpy as np
import logging
import pickle
import time
from tqdm import tqdm


logging.basicConfig(format="%(levelname)s - %(message)s", level=logging.INFO)


class NGram:
    """
    Args:
        n: The n of n-gram.
        token_path: The path to the token set.
    """
    def __init__(self, token_path, n=2):
        self.n = n
        self.head2tail = {}
        self.token_count = {}
        self.model = {'head2tail': self.head2tail, 'token_count': self.token_count, 'n': self.n}
        self.token_path = token_path
        self.int2token, self.token2int = load_token(self.token_path)
        self.default_token = '*'

    def load(self, path):
        """Load model from the model path.
        """
        self.model = pickle.load(open(path, 'rb'))
        self.head2tail = self.model['head2tail']
        self.token_count = self.model['token_count']
        self.n = self.model['n']

        logging.info(f"Loaded model from {path}")
        logging.info(f"Model info:\n"
                     f"\tn: {self.n}\n"
                     f"\thead2tail length: {len(self.model['head2tail'])}\n"
                     f"\ttokens: {len(self.model['token_count'])}")

    def save(self, path):
        """Save model to the model path.
        """
        self.model = {'head2tail': self.head2tail, 'token_count': self.token_count, 'n': self.n}
        pickle.dump(self.model, open(path, 'ab'))
        logging.info(f"Saved model at {path}")

    def predict_next_token(self, text, delimiter=''):
        """Given a text, predict the next token by the n-gram model.
        """
        if not delimiter:
            text = list(text)
        else:
            text = text.split(delimiter)

        assert len(text) >= self.n - 1, 'The length of the text must >= (n - 1)'

        head = text[- (self.n - 1):]
        head = ''.join(head)

        if head in self.head2tail:
            idx = np.argmax(self.head2tail[head])
            return self.int2token[idx]
        else:
            return self.default_token

    def topk_token_candidates(self, text, k):
        pass

    def fix(self, fp, delimiter='', anew=True):
        """Train a n-gram model.
        Args:
            fp: The path to the corpus.
            delimiter: Delimiter for splitting the text lines.
            anew: Train a new model, or train based on the existed model.
        """
        if anew:
            self.head2tail = {}
            self.token_count = {}

        logging.info(f"Begin training a {self.n}-gram model.")
        st = time.time()
        skipped = 0
        n_texts = 0
        n_tokens = 0
        with open(fp) as f:
            for line in f:
                if not delimiter:
                    line = list(line)
                else:
                    line = line.split(delimiter)

                line = [tk for tk in line if tk in self.token2int]  # Remove tokens that not in self.token2int

                if len(line) >= self.n:
                    for tokens in zip(*[line[i:] for i in range(self.n)]):
                        head = ''.join(tokens[:self.n - 1])
                        tail = tokens[self.n - 1]
                        if head in self.head2tail:
                            self.head2tail[head][self.token2int[tail]] += 1
                        else:
                            self.head2tail[head] = np.zeros((len(self.token2int),))
                            self.head2tail[head][self.token2int[tail]] = 1

                        for tk in tokens:
                            if tk in self.token_count:
                                self.token_count[tk] += 1
                            else:
                                self.token_count[tk] = 1
                    n_texts += 1
                    n_tokens += len(line)
                else:
                    skipped += 1

        logging.info(f"Finished training in {time.time() - st} seconds. {skipped} text lines were skipped.")
        logging.info(f"Trained with {n_texts} text lines ({n_tokens} tokens). "
                     f"The lengths of head2tail and token_count are "
                     f"{len(self.head2tail)} and {len(self.token_count)}.")

    def fix_v2(self, fp, delimiter='', anew=True):
        """Train a n-gram model. The corpus will be loaded into the memory all at once.
        Args:
            fp: The path to the corpus.
            delimiter: Delimiter for splitting the text lines.
            anew: Train a new model, or train based on the existed one.
        """
        if anew:
            self.head2tail = {}
            self.token_count = {}

        logging.info(f"Reading corpus.")
        lines = open(fp).readlines()
        logging.info(f"Read {len(lines)} text lines.")

        logging.info(f"Begin training a {self.n}-gram model.")
        st = time.time()
        skipped = 0
        n_texts = 0
        n_tokens = 0

        for line in tqdm(lines):
            if not delimiter:
                line = list(line)
            else:
                line = line.split(delimiter)

            line = [tk for tk in line if tk in self.token2int]  # Remove tokens that not in self.token2int

            if len(line) >= self.n:
                for tokens in zip(*[line[i:] for i in range(self.n)]):
                    head = ''.join(tokens[:self.n - 1])
                    tail = tokens[self.n - 1]
                    if head in self.head2tail:
                        self.head2tail[head][self.token2int[tail]] += 1
                    else:
                        self.head2tail[head] = np.zeros((len(self.token2int),))
                        self.head2tail[head][self.token2int[tail]] = 1

                    for tk in tokens:
                        if tk in self.token_count:
                            self.token_count[tk] += 1
                        else:
                            self.token_count[tk] = 1
                n_texts += 1
                n_tokens += len(line)
            else:
                skipped += 1

        logging.info(f"Finished training in {time.time() - st} seconds. {skipped} text lines were skipped.")
        logging.info(f"Trained with {n_texts} text lines ({n_tokens} tokens). "
                     f"The lengths of head2tail and token_count are "
                     f"{len(self.head2tail)} and {len(self.token_count)}.")


def load_token(fp):
    """Load tokens for the n-gram model.
    Args:
        fp: The path to the token set.
    Returns:
        int2token: A dict, mapping from integers to tokens.
        token2int: A dict, mapping from tokens to integers.
    """
    int2token, token2int = {}, {}
    with open(fp) as f:
        for i, line in enumerate(f):
            line = line.strip('\n ')
            int2token[i] = line
            token2int[line] = i
    return int2token, token2int


