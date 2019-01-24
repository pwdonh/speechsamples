#!/bin/bash
#
#$ -o ./out.txt
#$ -e ./err.txt
#$ -m e

source ~/.bashrc
cd $LOCAL/Projects/speechsamples/voxceleb_exp/

python -u train.py --cuda --basepath $LOCAL --savefile ./exp/v2/state_dict.pkl --model-type VoxResNetVAE > ./exp/v2/train.log
