# cd speechsamples/voxceleb_exp
from models import *
from data import *
import torch

import argparse

parser = argparse.ArgumentParser(description='DeepSpeech training')
parser.add_argument('--basepath', metavar='DIR',
        help='path to train manifest csv', default='/meg/meg1/users/peterd/')
parser.add_argument('--cuda', dest='cuda', action='store_true', help='Use cuda to train model')
parser.add_argument('--savefile', metavar='DIR',
        help='path to test manifest csv', default='./exp/state_dict.pkl')
args = parser.parse_args()

import os
# savefile = './exp/v1/state_dict.pkl'
savefile = args.savefile
# basepath = '/meg/meg1/users/peterd/'
basepath = args.basepath
audio_conf = {'sample_rate': 16000, 'window_size': .025, 'window_stride': .010, 'window': 'hamming'}

checkpoint_load = torch.load(savefile)
# checkpoint_load['fc_mu.weight'] = torch.eye(512).cuda()
# checkpoint_load['fc_mu.bias'] = torch.zeros(512).cuda()
embed_size = checkpoint_load['fc_mu.weight'].shape[0]
if 'fc_var.weight' in checkpoint_load:
    model_type = 'VoxResNetVAE'
else:
    model_type = 'VoxResNet'

model = voxresnet34(model_type, embed_size)
if args.cuda:
    model.cuda()
    device = torch.device("cuda:0")
else:
    device = torch.device("cpu")

model.load_state_dict(checkpoint_load)

test_manifest = './data/verification_test_all.csv'

test_dataset = SpectrogramVerificationDataset(audio_conf, test_manifest, basepath)
test_sampler = BucketingSampler(test_dataset, batch_size=64)
test_loader = AudioDataLoader(test_dataset, num_workers=1, batch_sampler=test_sampler)

def compute_eer(same, similarity):
    diffs = []
    rates = []
    for thresh in np.arange(.5,.99,.001):
        a=sum((same==1) & ((similarity>thresh)==False))/sum(same==1)
        b=sum((same==0) & ((similarity>thresh)==True))/sum(same==0)
        diffs.append(abs(a-b))
        rates.append(np.mean([a,b]))
        # print('{}: {} {}'.format(thresh,a,b))
    return rates[np.argmin(diffs)]

model.eval()
sim = torch.nn.CosineSimilarity()

similarity = []
same = []

for i, data in enumerate(test_loader):
    data = (data[0].to(device), data[1].to(device), data[2].to(device))
    print('{} of {}'.format(i, len(test_loader)))
    same += list(data[2].data.cpu().numpy())
    out1 = model.trunk(data[0])
    out2 = model.trunk(data[1])
    similarity += list(sim(out1, out2).data.cpu().numpy())
    if (i>0) and (i%10==0):
        print(compute_eer(np.array(same), np.array(similarity)))



#
# import seaborn
import matplotlib.pyplot as plt
plt.hist(similarity[same==0], alpha=.5)
plt.hist(similarity[same==1], alpha=.5)
#
# import pandas as pd
# pd.DataFrame({'different': similarity[same==0], 'same': similarity[same==1]})
#
# tips = seaborn.load_dataset("tips")


print('yes')
