#!/usr/bin/env python
# coding: utf-8

# In[27]:


import torchaudio
import glob
import numpy as np
import pandas as pd
from torch import nn
import torch.nn.functional as F
import torch

from skorch import NeuralNetClassifier
from torch.utils.data import Dataset, DataLoader
from skorch.helper import predefined_split
from sklearn.model_selection import train_test_split
import tqdm

import sklearn
from skorch.callbacks import EpochScoring, LRScheduler, Checkpoint
from torch.optim import Adam, SGD
from torch.optim.lr_scheduler import CyclicLR, CosineAnnealingLR

# In[3]:


f_all = glob.glob('../data/train/*/*')
print('len(f_all)=', len(f_all))

all_name = list(set([file.split('/')[-2] for file in f_all]))
name2idx = {name: idx for idx, name in enumerate(all_name)}
print(name2idx)
idx2name = {name2idx[key]: key for key in name2idx.keys()}
print(idx2name)

# In[4]:


labels = []
for filename in f_all:
    label = name2idx[filename.split('/')[-2]]
    labels.append(label)

# In[5]:


tra_files, val_files, tra_labels, val_labels = train_test_split(f_all, labels, test_size=0.33, random_state=42,
                                                                stratify=labels)


# In[21]:


def load(filename):
    waveform, sample_rate = torchaudio.load(filename)

    # new_sample_rate = 25000
    # waveform = torchaudio.transforms.Resample(sample_rate, new_sample_rate)(waveform)
    # waveform = self.normalize(waveform)

    pad_len_tresh = sample_rate * 1
    data = torch.zeros([pad_len_tresh])
    if waveform.size(1) >= pad_len_tresh:
        data = waveform[0, :pad_len_tresh]
    else:
        data[:waveform.size(1)] = waveform
    return data


class MyDataset(Dataset):
    def __init__(self, files, labels):
        super().__init__()
        self.files = files
        self.labels = labels

    def __len__(self):
        return len(self.files)

    def _get_label(self, index):
        return self.labels[index]

    def normalize(self, tensor):
        # 减去均值, 除以绝对值，缩放到[-1,1]
        tensor_minusmean = tensor - tensor.mean()
        return tensor_minusmean / tensor_minusmean.abs().max()

    def __getitem__(self, index):
        filename = self.files[index]
        data = load(filename)
        label = self._get_label(index)
        return data, label


tra = MyDataset(tra_files, tra_labels)
print(len(tra))
val = MyDataset(val_files, val_labels)
print(len(val))


# In[22]:


class DownSample2x(nn.Sequential):
    def __init__(self, _in, _out):
        super().__init__(
            nn.Conv1d(_in, _out, kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
        )


class SELayer(nn.Module):
    def __init__(self, _in, _hidden=64):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(_in, _hidden),
            nn.PReLU(),
            nn.Linear(_hidden, _in),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1)
        return x * y


class ResConv1d(nn.Module):
    def __init__(self, _in, _out):
        super(ResConv1d, self).__init__()

        self.cal = nn.Sequential(
            nn.Conv1d(_in, _out, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm1d(_out),
            nn.ReLU(),
            nn.Conv1d(_out, _out, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm1d(_out),
        )
        self.se = SELayer(_out, _out)
        self.conv = nn.Conv1d(_in, _out, kernel_size=1, padding=0, stride=1)
        self.relu = nn.ReLU()
        self.bn = nn.BatchNorm1d(_out)

    def forward(self, x):
        res = self.cal(x)
        res = self.se(res)

        x = self.bn(self.conv(x))

        return self.relu(res + x)


class MyModule(nn.Module):
    def __init__(self, nonlin=F.relu):
        super(MyModule, self).__init__()

        self.d1 = DownSample2x(201, 256)
        self.c1 = ResConv1d(256, 256)

        self.d2 = DownSample2x(256, 256)
        self.c2 = ResConv1d(256, 256)

        self.d3 = DownSample2x(256, 128)
        self.c3 = ResConv1d(128, 128)

        self.d4 = DownSample2x(128, 128)
        self.c4 = ResConv1d(128, 128)

        self.dropout = nn.Dropout(0.5)
        self.cls = nn.Linear(640, 30)

        self.t = torchaudio.transforms.Spectrogram()

    def preprocess(self, x, p=2, eps=1e-8):
        x = x / (x.norm(p=p, dim=1, keepdim=True) + eps)
        x = x.unsqueeze(1)
        return x

    def forward(self, x):
        bs = x.size(0)
        x = self.t(x)
        #         x = self.preprocess(x)

        x = self.d1(x)
        x = self.c1(x)

        x = self.d2(x)
        x = self.c2(x)

        x = self.d3(x)
        x = self.c3(x)

        x = self.d4(x)
        x = self.c4(x)

        x = x.reshape(bs, -1)
        x = self.dropout(x)

        return F.softmax(self.cls(x))


def microf1(net, ds, y=None):
    y_true = [y for _, y in ds]
    y_pred = net.predict(ds)
    return sklearn.metrics.f1_score(y_true, y_pred, average='micro')


def macrof1(net, ds, y=None):
    y_true = [y for _, y in ds]
    y_pred = net.predict(ds)
    return sklearn.metrics.f1_score(y_true, y_pred, average='macro')


# In[28]:


net = NeuralNetClassifier(
    MyModule,
    max_epochs=100,
    lr=0.001,
    batch_size=1024,
    optimizer=Adam,
    iterator_train__shuffle=True,
    iterator_train__num_workers=4,
    iterator_train__pin_memory=True,
    train_split=predefined_split(val),
    callbacks=[LRScheduler(policy=CosineAnnealingLR, T_max=64),
               EpochScoring(macrof1, use_caching=True, lower_is_better=False),
               EpochScoring(microf1, use_caching=True, lower_is_better=False),
               Checkpoint(monitor='macrof1_best', dirname='model')],
    device='cuda',
    verbose=1
)

print('start training')
_ = net.fit(tra, y=None)
# net.initialize()
net.load_params(f_params='model/params.pt', f_optimizer='model/optimizer.pt', f_history='model/history.json')

# In[ ]:


submission = pd.read_csv('../data/submission.csv')
ann_prob = np.zeros(len(submission))
ann_labels = []

for i in tqdm.tqdm(range(len(submission))):
    ann_row = submission.iloc[i]
    ann_fname = '../data/test/{}'.format(ann_row['file_name'])
    ann_wav = load(ann_fname)  # 读取annotation中的样本
    ann_wav = torch.unsqueeze(ann_wav, dim=0)
    score = net.predict_proba(ann_wav)  # 两个向量的内积
    ann_prob[i] = np.argmax(score)
    ann_labels.append(idx2name[ann_prob[i]])

submission['label'] = ann_labels
submission.to_csv('result.csv')
