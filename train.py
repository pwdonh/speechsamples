# pwd
# cd '/local_raid/data/peterd/Projects/speechsamples/voxceleb_exp'
from models import *
from data import *
from torch import optim
from time import time
# import matplotlib.pyplot as plt
import argparse

parser = argparse.ArgumentParser(description='DeepSpeech training')
parser.add_argument('--model-type', metavar='DIR',
        help='path to train manifest csv', default='VoxResNet')
parser.add_argument('--train-manifest', metavar='DIR',
        help='path to train manifest csv', default='/meg/meg1/users/peterd/data/Language/dev_mp3/identification_train.csv')
parser.add_argument('--test-manifest', metavar='DIR',
        help='path to test manifest csv', default='/meg/meg1/users/peterd/data/Language/dev_mp3/identification_test.csv')
parser.add_argument('--savefile', metavar='DIR',
        help='path to test manifest csv', default='./exp/state_dict.pkl')
parser.add_argument('--cuda', dest='cuda', action='store_true', help='Use cuda to train model')
args = parser.parse_args()

torch.manual_seed(123456)
torch.cuda.manual_seed_all(123456)

# train_manifest = '/meg/meg1/users/peterd/data/Language/dev_mp3/identification_train.csv'
# test_manifest = '/meg/meg1/users/peterd/data/Language/dev_mp3/identification_test.csv'
# model_type = 'VoxResNetVAE'
# exec('model_class = '+model_type)
exec('model_class = '+args.model_type)
train_manifest = args.train_manifest
test_manifest = args.test_manifest
savefile = args.savefile
audio_conf = {'sample_rate': 16000, 'window_size': .025, 'window_stride': .010, 'window': 'hamming'}
batch_size = 64

train_dataset = SpectrogramDataset(audio_conf, train_manifest)
train_sampler = BucketingSampler(train_dataset, batch_size=batch_size)
train_loader = AudioDataLoader(train_dataset, num_workers=1, batch_sampler=train_sampler)
test_dataset = SpectrogramDataset(audio_conf, test_manifest)
test_sampler = BucketingSampler(test_dataset, batch_size=batch_size)
test_loader = AudioDataLoader(test_dataset, num_workers=1, batch_sampler=test_sampler)

model = voxresnet34(model_class)
if args.cuda:
    model.cuda()
    device = torch.device("cuda:0")
else:
    device = torch.device("cpu")

optimizer = optim.Adam(model.parameters())
n_batch = int(len(train_dataset)/batch_size)
best_val_loss = 0.

for epoch in range(30):

    model.train()
    avg_loss = 0.
    epoch_start = time()
    train_sampler.shuffle(epoch)
    for i, (data) in enumerate(train_loader, start=0):

        data = (data[0].to(device), data[1].to(device))

        # Prediction
        out = model(data[0])
        loss = model.loss(out, data[1])

        # compute gradient
        optimizer.zero_grad()
        loss.backward()

        print(max(p.grad.data.abs().max() for p in model.parameters()))
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)

        # torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_norm)
        optimizer.step()

        print('Batch {} of {}, {}'.format(i, n_batch, loss.item()))

        loss_eval = model.loss_eval(out, data[1])
        avg_loss += loss_eval.item()

        if (i>0) and (i%10==0):
            print('Average training loss: {}'.format(avg_loss/i))
            epoch_time = time()-epoch_start
            print('Epoch {}: {} of {} minutes'.format(epoch, epoch_time/60, ((epoch_time/i)*(n_batch))/60))

    # Evaluate test loss
    model.eval()
    avg_test_loss = 0.
    for i, (data) in enumerate(test_loader, start=0):
        data = (data[0].to(device), data[1].to(device))
        # Prediction
        out = model(data[0])
        loss = model.loss_eval(out, data[1])
        avg_test_loss += loss.item()
        if i>0:
            print(avg_test_loss/i)

    print('Test loss: {}'.format(avg_test_loss/i))

    if (epoch==0) or (avg_test_loss/i < best_val_loss):
        best_val_loss = avg_test_loss/i
        with open(savefile, 'wb') as f:
            torch.save(model.state_dict(), f)
