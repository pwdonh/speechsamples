# pwd
# cd speechsamples/voxceleb_exp
import os, sys
import torchaudio
import sys
import numpy as np
import argparse
# sys.path.append('/meg/meg1/users/peterd/Projects/speechsamples/deepspeech.pytorch/data')
from data_loader import *

parser = argparse.ArgumentParser(description='DeepSpeech training')
parser.add_argument('--basepath', metavar='DIR',
        help='path to train manifest csv', default='/meg/meg1/users/peterd/')
parser.add_argument('--cuda', dest='cuda', action='store_true', help='Use cuda to train model')
args = parser.parse_args()

basepath = args.basepath
# basepath = os.environ['MEGDISK']

train_dir = os.path.join(basepath, 'data/Language/dev_mp3/aac')
pid_list = os.listdir(train_dir)
len_data = np.zeros(len(pid_list))

train_file = os.path.join(basepath, 'data/Language/dev_mp3/identification_train.csv')
test_file = os.path.join(basepath, 'data/Language/dev_mp3/identification_test.csv')
f_train = open(train_file, 'w')
f_test = open(test_file, 'w')
f_train.close()
f_test.close()

def write_video_to_file(f):
    video_dir = os.path.join(pid_dir, video)
    for file in os.listdir(video_dir):
        filepath = os.path.join(video_dir, file)
        # filepath_short = filepath.split(os.path.join(basepath, 'data/Language/dev_mp3/aac/'))[1]
        audiolen = get_audio_length(filepath)
        if audiolen>3:
            line = filepath+','+str(i_pid)+','+str(audiolen)+'\n'
            f.write(line)

for i_pid, pid in enumerate(pid_list):

    print('ID {} of {}'.format(i_pid, len(pid_list)))

    f_train = open(train_file, 'a')
    f_test = open(test_file, 'a')
    pid_dir = os.path.join(train_dir, pid)
    videos = os.listdir(pid_dir)
    # randomly select test video
    test_video = np.random.randint(len(videos))

    for i_video, video in enumerate(videos):
        if i_video==test_video:
            write_video_to_file(f_test)
        else:
            write_video_to_file(f_train)

    f_train.close()
    f_test.close()


    # if False:
    #
    #     for video in videos:
    #         video_dir = os.path.join(pid_dir, video)
    #         len_data[i_pid] += sum([get_audio_length(os.path.join(video_dir,f)) for f in os.listdir(video_dir)])
    #
    #     print(pid)
    #     print(len_data[i_pid])
