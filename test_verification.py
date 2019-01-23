# cd speechsamples/voxceleb_exp
from models import *
from data import *
import torch

import argparse

parser = argparse.ArgumentParser(description='DeepSpeech training')
parser.add_argument('--basepath', metavar='DIR',
        help='path to train manifest csv', default='/meg/meg1/users/peterd/')
parser.add_argument('--cuda', dest='cuda', action='store_true', help='Use cuda to train model')
args = parser.parse_args()

model = voxresnet34(VoxResNet)
if args.cuda:
    model.cuda()
    device = torch.device("cuda:0")
else:
    device = torch.device("cpu")

import os
savefile = './exp/v1/state_dict.pkl'
basepath = args.basepath
audio_conf = {'sample_rate': 16000, 'window_size': .025, 'window_stride': .010, 'window': 'hamming'}

checkpoint_load = torch.load(savefile)
model.load_state_dict(checkpoint_load)

test_manifest = './data/verification_test_all.csv'

test_dataset = SpectrogramVerificationDataset(audio_conf, test_manifest, basepath)
test_sampler = BucketingSampler(test_dataset, batch_size=1)
test_loader = AudioDataLoader(test_dataset, num_workers=1, batch_sampler=test_sampler)

model.eval()
sim = torch.nn.CosineSimilarity()

similarity = []
same = []

for i, data in enumerate(test_loader):
    data = (data[0].to(device), data[1].to(device))
    print(i)
    same.append( data[2].item() )
    out1 = model(data[0])
    out2 = model(data[1])
    similarity.append( sim(out1, out2).item() )

same = np.array(same)
similarity = np.array(similarity)

diffs = []
rates = []
for thresh in np.arange(.5,.99,.001):
    a=sum((same==1) & ((similarity>thresh)==False))/sum(same==1)
    b=sum((same==0) & ((similarity>thresh)==True))/sum(same==0)
    diffs.append(abs(a-b))
    rates.append(np.mean([a,b]))
    # print('{}: {} {}'.format(thresh,a,b))
rates[np.argmin(diffs)]

#
# import seaborn
# import matplotlib.pyplot as plt
# plt.hist(similarity[same==0])
# plt.hist(similarity[same==1])
#
# import pandas as pd
# pd.DataFrame({'different': similarity[same==0], 'same': similarity[same==1]})
#
# tips = seaborn.load_dataset("tips")


print('yes')
