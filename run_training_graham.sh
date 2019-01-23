#!/bin/bash
#SBATCH --time=24:00:00
#SBATCH --account=def-denisek
#SBATCH --gres=gpu:1
#SBATCH --mem=4000M

mkdir $SLURM_TMPDIR/data
mkdir $SLURM_TMPDIR/data/Language
# mkdir /tmp/data/Language/dev_mp3
# mkdir /tmp/data/Language/dev_mp3/aac/

rsync -r $LOCAL/data/Language/dev_mp3 $SLURM_TMPDIR/data/Language/

# cp -r $LOCAL/data/Language/dev_mp3/aac/id00* /tmp/data/Language/dev_mp3/aac &
# cp -r $LOCAL/data/Language/dev_mp3/aac/id01* /tmp/data/Language/dev_mp3/aac &
# cp -r $LOCAL/data/Language/dev_mp3/aac/id02* /tmp/data/Language/dev_mp3/aac &
# wait
# cp -r $LOCAL/data/Language/dev_mp3/aac/id03* /tmp/data/Language/dev_mp3/aac &
# cp -r $LOCAL/data/Language/dev_mp3/aac/id04* /tmp/data/Language/dev_mp3/aac &
# cp -r $LOCAL/data/Language/dev_mp3/aac/id05* /tmp/data/Language/dev_mp3/aac &
# wait
# cp -r $LOCAL/data/Language/dev_mp3/aac/id06* /tmp/data/Language/dev_mp3/aac &
# cp -r $LOCAL/data/Language/dev_mp3/aac/id07* /tmp/data/Language/dev_mp3/aac &
# wait
# cp -r $LOCAL/data/Language/dev_mp3/aac/id08* /tmp/data/Language/dev_mp3/aac &
# cp -r $LOCAL/data/Language/dev_mp3/aac/id09* /tmp/data/Language/dev_mp3/aac &
# wait
# cp $LOCAL/data/Language/dev_mp3/identification_train.csv /tmp/data/Language/dev_mp3/identification_train.csv
# cp $LOCAL/data/Language/dev_mp3/identification_test.csv $SLURM_TMPDIRIR/data/Language/dev_mp3/identification_test.csv

# export fcount=0
# until [ $fcount \> 5995 ]; do fcount=$(ls $SLURM_TMPDIRIR/data/Language/dev_mp3/aac/ | wc -l); done

sed -i "s|/tmp/|/$SLURM_TMPDIR/|g" $SLURM_TMPDIR/data/Language/dev_mp3/identification_train.csv
sed -i "s|/tmp/|/$SLURM_TMPDIR/|g" $SLURM_TMPDIR/data/Language/dev_mp3/identification_test.csv
# sed -i -e 's/home\/peterd\/project\/peterd/$SLURM_TMPDIR/g' $SLURM_TMPDIR/data/Language/dev_mp3/identification_train.csv
# sed -i -e 's/home\/peterd\/project\/peterd/$SLURM_TMPDIR/g' $SLURM_TMPDIR/data/Language/dev_mp3/identification_test.csv
# sed -i -e 's/meg\/meg1\/users\/peterd/$SLURM_TMPDIR/g' $SLURM_TMPDIR/data/Language/dev_mp3/identification_train.csv
# sed -i -e 's/meg\/meg1\/users\/peterd/$SLURM_TMPDIR/g' $SLURM_TMPDIR/data/Language/dev_mp3/identification_test.csv

cd /home/peterd/project/peterd/Projects/speechsamples/voxceleb_exp/
source venv/bin/activate
python -u train.py --cuda --train-manifest $SLURM_TMPDIR/data/Language/dev_mp3/identification_train.csv --test-manifest $SLURM_TMPDIR/data/Language/dev_mp3/identification_test.csv > train_graham.log
