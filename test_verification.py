# cd speechsamples/voxceleb_exp
from models import *
from data import *
import torch

model = voxresnet34(VoxResNetAdaptive)
if args.cuda:
    model.cuda()
    device = torch.device("cuda:0")
else:
    device = torch.device("cpu")

import os
savefile = os.environ['MEGDISK']+'/Projects/speechsamples/voxceleb_exp/exp/v1/state_dict.pkl'
audio_conf = {'sample_rate': 16000, 'window_size': .025, 'window_stride': .010, 'window': 'hamming'}

checkpoint_load = torch.load(savefile)
model.load_state_dict(checkpoint_load)

test_manifest = os.path.join(os.environ['LOCAL'], 'data/Language/voxceleb1/verification_test_all.csv')

test_dataset = SpectrogramVerificationDataset(audio_conf, test_manifest)
test_sampler = BucketingSampler(test_dataset, batch_size=1)
test_loader = AudioDataLoader(test_dataset, num_workers=1, batch_sampler=test_sampler)

# model = voxresnet34()
sim = torch.nn.CosineSimilarity()

similarity = []
same = []

for i, data in enumerate(test_loader):
    print(i)
    same.append( data[2].item() )
    out1 = model(data[0])
    out2 = model(data[1])
    similarity.append( sim(out1, out2).item() )
    # if i==10:
    #     break

import pdb; pdb.set_trace()

same = np.array(same)
similarity = np.array(similarity)

for thresh in np.arange(.5,.99,.01):
    a=sum(((same>0)==True) & ((similarity>thresh)==False))
    b=sum(((same>0)==False) & ((similarity>thresh)==True))
    if a==b:
        print(thresh)
    print('{} {}'.format(a,b))

import matplotlib.pyplot as plt
plt.hist(similarity[same==0])
plt.hist(similarity[same==1])



print('yes')
