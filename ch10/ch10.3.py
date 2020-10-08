import collections
import math
import random
import sys
import time
import os
import numpy as np
import torch
from torch import nn
import torch.utils.data as Data

sys.path.append("..")
import d2lzh_pytorch as d2l
print(torch.__version__)

# 处理数据集
assert 'ptb.train.txt' in os.listdir("data/ptb")

with open('data/ptb/ptb.train.txt', 'r') as f:
    lines = f.readlines()
    # st是sentence的缩写
    raw_dataset = [st.split() for st in lines]

'# sentences: %d' % len(raw_dataset) # 输出 '# sentences: 42068'

# 对于数据集的前3个句子，打印每个句子的词数和前5个词。这个数据集中句尾符为"<eos>"，生僻词全用"<unk>"表示，数字则被替换成了"N"。

for st in raw_dataset[:3]:
    print('# tokens:', len(st), st[:5])

# 建立词语索引
# tk是token的缩写  为了计算简单，我们只保留在数据集中至少出现5次的词
counter = collections.Counter([tk for st in raw_dataset for tk in st])
counter = dict(filter(lambda x: x[1] >= 5, counter.items()))
# 然后将词映射到整数索引
idx_to_token = [tk for tk, _ in counter.items()]
token_to_idx = {tk: idx for idx, tk in enumerate(idx_to_token)}
dataset = [[token_to_idx[tk] for tk in st if tk in token_to_idx]
           for st in raw_dataset]
num_tokens = sum([len(st) for st in dataset])
'# tokens: %d' % num_tokens # 输出 '# tokens: 887100'

# 对词进行二次采样
def discard(idx):
    return random.uniform(0, 1) < 1 - math.sqrt(
        1e-4 / counter[idx_to_token[idx]] * num_tokens)

subsampled_dataset = [[tk for tk in st if not discard(tk)] for st in dataset]
'# tokens: %d' % sum([len(st) for st in subsampled_dataset]) # '# tokens: 375875'

# 比较一个词在二次采样前后出现在数据集中的次数。可见高频词“the”的采样率不足1/20
def compare_counts(token):
    return '# %s: before=%d, after=%d' % (token, sum(
        [st.count(token_to_idx[token]) for st in dataset]), sum(
        [st.count(token_to_idx[token]) for st in subsampled_dataset]))

print(compare_counts('the')) # '# the: before=50770, after=2013'

# 提取中心词和背景词
def get_centers_and_contexts(dataset, max_window_size):
    centers, contexts = [], []
    for st in dataset:
        if len(st) < 2:  # 每个句子至少要有2个词才可能组成一对“中心词-背景词”
            continue
        centers += st
        for center_i in range(len(st)):
            window_size = random.randint(1, max_window_size)
            indices = list(range(max(0, center_i - window_size),
                                 min(len(st), center_i + 1 + window_size)))
            indices.remove(center_i)  # 将中心词排除在背景词之外
            contexts.append([st[idx] for idx in indices])
    return centers, contexts

tiny_dataset = [list(range(7)), list(range(7, 10))]
print('dataset', tiny_dataset)
for center, context in zip(*get_centers_and_contexts(tiny_dataset, 2)):
    print('center', center, 'has contexts', context)

all_centers, all_contexts = get_centers_and_contexts(subsampled_dataset, 5)
