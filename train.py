import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import tensor, LongTensor
from torch.utils.data import DataLoader
from bert_embedding import BertEmbedding
from pickle import load
from pickle import dump
from collections import Counter
import argparse
bert_embedding = BertEmbedding(max_seq_length=75)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def load_clean_sentences(filename):
    return load(open(filename, 'rb'))


def bert_to_tensor(result):
    embeds = result[0][1]
    tens = torch.empty(len(embeds), 768)
    idx = 0
    for k in embeds:
        tens[idx] = tensor(k)
        idx += 1
    return tens


class Dataset:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __getitem__(self, idx):
        en_sequence = self.x[idx]
        fr_sequence = self.y[idx]
        return en_sequence, fr_sequence

    def __len__(self):
        return len(self.x)


def seq2seq_collate_bert(samples, pad_first=False, backwards=False):
    pad_idx = 0
    max_len_x, max_len_y = max([len(s[0].split()) for s in samples]), max([len(s[1]) for s in samples])
    res_x = torch.zeros(len(samples), max_len_x, 768).float() + pad_idx
    res_y = torch.zeros(len(samples), max_len_y).long() + pad_idx
    for i, s in enumerate(samples):
        vec = bert_to_tensor(bert_embedding([s[0]]))
        res_x[i, :len(s[0].split())] = vec
        res_y[i, :len(s[1]):] = LongTensor(s[1])
    return res_x, res_y


def get_dls(dataset, collate_fn, bs):
    dataloader = DataLoader(dataset, bs, collate_fn=collate_fn, num_workers=1)
    return dataloader


def conv3x3(in_planes, out_planes, stride=1, groups=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, groups=groups, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


def seq2seq_loss(input, target):
    bs, sl = target.size()
    bs_in, sl_in, nc = input.size()
    if sl > sl_in:
        input = F.pad(input, (0, 0, 0, 0, 0, sl - sl_in))
    input = input[:, :sl]
    return F.cross_entropy(input.contiguous().view(-1, nc), target.view(-1))  # , ignore_index=1)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


def get_data(bs):
    en_subset_sentences_train = load_clean_sentences("en_train.pkl")
    en_subset_sentences_valid = load_clean_sentences("en_valid.pkl")
    fr_sentences_tokens_train = load_clean_sentences("fr_train.pkl")
    fr_sentences_tokens_valid = load_clean_sentences("fr_valid.pkl")

    train_dataset = Dataset(en_subset_sentences_train, fr_sentences_tokens_train)
    valid_dataset = Dataset(en_subset_sentences_valid, fr_sentences_tokens_valid)

    train_dl = get_dls(train_dataset, seq2seq_collate_bert, bs)
    valid_dl = get_dls(valid_dataset, seq2seq_collate_bert, bs)

    return train_dl, valid_dl


def train(model, train_dl, valid_dl, opt, epochs=1):
    for epoch in range(epochs):
        model.train()
        tot_tr_loss = 0
        for x, y in train_dl:
            x, y = x.to(device), y.to(device)
            input = model(x.unsqueeze(1))
            input = input.reshape(x.shape[0], 36, -1)
            loss = seq2seq_loss(input, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot_tr_loss += loss.item()
            print(loss.item())
        model.eval()
        with torch.no_grad():
            tot_valid_loss = 0
            for x, y in valid_dl:
                x, y = x.to(device), y.to(device)
                input = model(x.unsqueeze(1))
                input = input.reshape(x.shape[0], 36, -1)
                loss = seq2seq_loss(input, y)
                tot_valid_loss += loss.item()

        print(epoch, tot_tr_loss / len(train_dl), tot_valid_loss / len(valid_dl))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-bs", "--batch_size", type=int, help="batch_size", default=2)
    args = parser.parse_args()

    model = nn.Sequential(BasicBlock(1, 64),
                          nn.Conv2d(64, 512, 3, padding=1),
                          nn.AdaptiveMaxPool3d((1, 36, 58802))).to(device)

    opt = torch.optim.Adam(model.parameters())
    batch_size = args.batch_size
    train_dl, valid_dl = get_data(batch_size)
    print("before training", batch_size)
    train(model, train_dl, valid_dl, opt, epochs=1)


main()
