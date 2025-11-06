#!/bin/bash
METHOD=er
DATASET=imagenet100

#LAMBS=(0.3 1.0)
#THRESH=(0.3 0.5 0.7 0.9)
#GAMMAS=(0.3 0.5 0.7 0.9)
LAMB=0
THRESH=0
GAMMA=0

RANDS=(1 2 3)
GPU_IDX=0

for SEED in ${RANDS}
do
    NAME=exp_${DATASET}_disjoint
    echo $NAME
    ./scripts/exp_script_disjoint.sh ${NAME} ${METHOD} ${DATASET} ${SEED} ${GPU_IDX} 0 2000 4 ${THRESH} ${GAMMA} ${LAMB} > ./nohup_logs/${NAME}_${METHOD}_${SEED}.out
done

#   ./scripts/exp_script_disjoint.sh exp_debug er cifar100 1 0 0 2000 4 0 0 0 
