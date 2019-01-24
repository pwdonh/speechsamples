from pandas import read_csv
import os
from data_loader import *

all_file = os.path.join(os.environ['LOCAL'], 'data/Language/voxceleb1/verification_test_all_tmp.csv')
hard_file = os.path.join(os.environ['LOCAL'], 'data/Language/voxceleb1/verification_test_hard_tmp.csv')
f_all = open(all_file, 'w')
f_hard = open(hard_file, 'w')
f_all.close()
f_hard.close()

basepath = os.path.join(os.environ['LOCAL'], 'data/Language/voxceleb1/mp3/')

def write_to_file(table, filename):
    for filepath1, filepath2, same in zip(table[1], table[2], table[0]):
        filepath1 = basepath+filepath1.replace('.wav','.mp3')
        filepath2 = basepath+filepath2.replace('.wav','.mp3')
        audiolen1 = get_audio_length(filepath1)
        audiolen2 = get_audio_length(filepath2)
        f = open(filename, 'a')
        line = filepath1+','+filepath2+','+str(same)+','+str(audiolen1)+','+str(audiolen2)+'\n'
        f.write(line)
        f.close()

table = read_csv(os.environ['LOCAL']+'data/Language/voxceleb1/meta/list_test_all.txt', sep=' ', header=None)
write_to_file(table, all_file)

table = read_csv(os.environ['LOCAL']+'data/Language/voxceleb1/meta/list_test_hard.txt', sep=' ', header=None)
write_to_file(table, hard_file)
