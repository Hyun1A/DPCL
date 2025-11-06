#!/bin/bash
METHOD=er
DATASET=cifar100_sc

#LAMBS=(0.3 1.0)
#THRESH=(0.3 0.5 0.7 0.9)
#GAMMAS=(0.3 0.5 0.7 0.9)
LAMB=0
THRESH=0
GAMMA=0

SEEDS=(1 2 3)
GPU_IDX=0

for SEED in ${SEEDS}
do
    NAME=exp_${DATASET}_disjoint
    echo $NAME
    ./scripts/exp_script_disjoint.sh ${NAME} ${METHOD} ${DATASET} ${SEED} ${GPU_IDX} 0 2000 4 ${THRESH} ${GAMMA} ${LAMB} > ./nohup_logs/${NAME}_${METHOD}_${SEED}.out
done

for SEED in ${SEEDS}
do
    NAME=exp_${DATASET}_blurry
    echo $NAME
    ./scripts/exp_script_blurry.sh ${NAME} ${METHOD} ${DATASET} ${SEED} ${GPU_IDX} 0 2000 4 ${THRESH} ${GAMMA} ${LAMB} > ./nohup_logs/${NAME}_${METHOD}_${SEED}.out
done

for SEED in ${SEEDS}
do
    NAME=exp_${DATASET}_iblurry
    echo $NAME
    ./scripts/exp_script_iblurry.sh ${NAME} ${METHOD} ${DATASET} ${SEED} ${GPU_IDX} 0 2000 4 ${THRESH} ${GAMMA} ${LAMB} > ./nohup_logs/${NAME}_${METHOD}_${SEED}.out
done