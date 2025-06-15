#!/usr/bin/env python
import os
import string
import random
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import torch
import torch.nn as nn
import torch.nn.functional as F

class CharTransformer(nn.Module):
    def __init__(self, vocab_size, embedding_dim=128, num_heads=4, hidden_dim=256, num_layers=2, max_len=100):
        super(CharTransformer, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.pos_embedding = nn.Parameter(torch.zeros(1, max_len, embedding_dim))

        encoder_layer = nn.TransformerEncoderLayer(d_model=embedding_dim, nhead=num_heads, dim_feedforward=hidden_dim, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.fc = nn.Linear(embedding_dim, vocab_size)

    def forward(self, x):
        seq_len = x.size(1)
        x = self.embedding(x) + self.pos_embedding[:, :seq_len, :]
        out = self.transformer(x)
        logits = self.fc(out[:, -1, :])  # Use the last token's output for prediction
        return logits, None

class MyModel:
    """
    This is a starter model to get you started. Feel free to modify this file.
    """
    sequence_length = 10
    vocab = sorted(set(string.printable))
    char2idx = {c: i for i, c in enumerate(vocab)}
    idx2char = {i: c for c, i in char2idx.items()}
    vocab_size = len(vocab)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def __init__(self,char2idx, idx2char, vocab_size):
        self.char2idx = char2idx
        self.idx2char = idx2char
        self.vocab_size = vocab_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = CharTransformer(self.vocab_size).to(self.device)

    @classmethod
    def load_training_data(cls):
        # your code here
        # this particular model doesn't train
        return []

    @classmethod
    def load_test_data(cls, fname):
        # your code here
        data = []
        with open(fname) as f:
            for line in f:
                inp = line[:-1]  # the last character is a newline
                data.append(inp)
        return data

    @classmethod
    def write_pred(cls, preds, fname):
         with open(fname, 'w', encoding='utf-8') as f:
            for p in preds:
                f.write('{}\n'.format(p))

    def run_train(self, data, work_dir):
        # your code here
        pass
    """
    def run_pred(self, data):
        self.model.eval()
        preds = []
        with torch.no_grad():
            for context in data:
                if isinstance(context, tuple):  # handle (context, target) tuple
                    context = context[0]
                x = torch.tensor([[self.char2idx.get(c, 0) for c in context]], dtype=torch.long).to(self.device)
                logits, _ = self.model(x)
                probs = torch.softmax(logits, dim=-1)
                topk = torch.topk(probs, 10)  # get more in case some are space

                filtered_chars = []
                for idx in topk.indices[0]:
                    ch = self.idx2char[idx.item()]
                    if ch not in (' ', '\n', '\t', '\r'):  # remove whitespace characters
                        filtered_chars.append(ch)
                    if len(filtered_chars) == 3:
                        break

                # pad with "?" if not enough characters found
                while len(filtered_chars) < 3:
                    filtered_chars.append('?')

                preds.append(''.join(filtered_chars))
        preds = self.clean_preds(preds)    
        return preds
        """
    def run_pred(self, data):
        self.model.eval()
        preds = []
        with torch.no_grad():
            for context in data:
                # Handle case where test data is a (context, next_char) tuple
                if isinstance(context, tuple):
                    context = context[0]

                # Convert input characters to indices
                x = torch.tensor([[self.char2idx.get(c, 0) for c in context]], dtype=torch.long).to(self.device)

                # Get model predictions
                temperature = 1.0  # try 0.8 or 1.2 as well
                logits, _ = self.model(x)
                #probs = torch.softmax(logits, dim=-1)
                probs = torch.softmax(logits / temperature, dim=-1)
                topk = torch.topk(probs, 10)  # get top-10 in case some are whitespace

                # Filter out whitespace and collect top 3 chars
                filtered_chars = []
                for idx in topk.indices[0]:
                    ch = self.idx2char[idx.item()]
                    if ch not in (' ', '\n', '\t', '\r') and ch not in filtered_chars:
                        filtered_chars.append(ch)
                    if len(filtered_chars) == 3:
                        break

                # Pad if not enough found
                while len(filtered_chars) < 3:
                    filtered_chars.append('?')

                preds.append(''.join(filtered_chars))

        # Optional deduplication / sanitization
        preds = self.clean_preds(preds)
        return preds

    def clean_preds(self, preds):
        cleaned = []
        banned_chars = set('e.,:')
        for pred in preds:
            # Remove banned characters, preserve up to 3 characters
            filtered = ''.join([c for c in pred if c not in banned_chars])[:3]
            # Pad if too short
            if len(filtered) < 3:
                filtered = filtered.ljust(3, 'x')  # pad with 'x' or any placeholder
            cleaned.append(filtered)
        return cleaned



    def save(self, work_dir):
        # your code here
        # this particular model has nothing to save, but for demonstration purposes we will save a blank file
        with open(os.path.join(work_dir, 'model.checkpoint'), 'wt') as f:
            f.write('dummy save')

    @classmethod
    def load(cls, work_dir):
        # your code here
        # this particular model has nothing to load, but for demonstration purposes we will load a blank file
        #with open(os.path.join(work_dir, 'model.checkpoint')) as f:
         #   dummy_save = f.read()
        #return MyModel()
        checkpoint_path = os.path.join(work_dir, 'model.pt')
        checkpoint = torch.load(checkpoint_path, map_location=cls.device)
        model = cls(
        char2idx=checkpoint['char2idx'],
        idx2char=checkpoint['idx2char'],
        vocab_size=checkpoint['vocab_size']
        )
        model.model.load_state_dict(checkpoint['model_state_dict'])
        model.model.to(cls.device)
        model.model.eval()
        return model


if __name__ == '__main__':
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('mode', choices=('train', 'test'), help='what to run')
    parser.add_argument('--work_dir', help='where to save', default='work')
    parser.add_argument('--test_data', help='path to test data', default='example/input.txt')
    parser.add_argument('--test_output', help='path to write test predictions', default='pred.txt')
    args = parser.parse_args()

    random.seed(0)

    if args.mode == 'train':
        if not os.path.isdir(args.work_dir):
            print('Making working directory {}'.format(args.work_dir))
            os.makedirs(args.work_dir)
        print('Instatiating model')
        model = MyModel()
        print('Loading training data')
        train_data = MyModel.load_training_data()
        print('Training')
        model.run_train(train_data, args.work_dir)
        print('Saving model')
        model.save(args.work_dir)
    elif args.mode == 'test':
        print('Loading model')
        model = MyModel.load(args.work_dir)
        print('Loading test data from {}'.format(args.test_data))
        test_data = MyModel.load_test_data(args.test_data)
        print('Making predictions')
        pred = model.run_pred(test_data)
        print('Writing predictions to {}'.format(args.test_output))
        assert len(pred) == len(test_data), 'Expected {} predictions but got {}'.format(len(test_data), len(pred))
        model.write_pred(pred, args.test_output)
    else:
        raise NotImplementedError('Unknown mode {}'.format(args.mode))
