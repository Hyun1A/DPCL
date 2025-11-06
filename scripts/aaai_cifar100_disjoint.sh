#!/bin/bash
METHOD=er
DATASET=cifar100

#LAMBS=(0.3 1.0)
#THRESH=(0.3 0.5 0.7 0.9)
#GAMMAS=(0.3 0.5 0.7 0.9)

LAMB=0
THRESH=0
GAMMA=0

SEEDS=(1)
GPU_IDX=10

for SEED in ${SEEDS}
do
    NAME=exp_${DATASET}_disjoint
    echo $NAME
    taskset -c 70 ./scripts/exp_script_disjoint.sh ${NAME} ${METHOD} ${DATASET} ${SEED} ${GPU_IDX} 0 2000 4 ${THRESH} ${GAMMA} ${LAMB} #> ./nohup_logs/${NAME}_${METHOD}_${SEED}.out
done