#!/bin/bash
#
#$ -o ./out.txt
#$ -e ./err.txt
#$ -m e

source ~/.bashrc
cd $LOCAL/Projects/speechsamples/voxceleb_exp/

python -u test_verification.py --cuda > ./exp/v1/test_verification.log
