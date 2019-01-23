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
savefile = os.environ['LOCAL']+'/Projects/speechsamples/voxceleb_exp/exp/v1/state_dict.pkl'
audio_conf = {'sample_rate': 16000, 'window_size': .025, 'window_stride': .010, 'window': 'hamming'}

checkpoint_load = torch.load(savefile)
model.load_state_dict(checkpoint_load)

test_manifest = os.path.join(os.environ['LOCAL'], 'data/Language/voxceleb1/verification_test_all.csv')

test_dataset = SpectrogramVerificationDataset(audio_conf, test_manifest)
test_sampler = BucketingSampler(test_dataset, batch_size=1)
test_loader = AudioDataLoader(test_dataset, num_workers=1, batch_sampler=test_sampler)

model.eval()
sim = torch.nn.CosineSimilarity()

similarity = []
same = []

for i, data in enumerate(test_loader):
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
