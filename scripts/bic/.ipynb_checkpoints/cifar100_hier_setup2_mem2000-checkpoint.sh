#!/bin/bash
METHOD=bic
DATASET=cifar100_hier_setup2

SEEDS=(0 1 2)
GPU_IDX=0
DEBUG=0
MEM_SIZE=2000

DIR=./logs/${DATASET}/${METHOD}

if ! [ -d "./logs/${DATASET}" ]; then
    mkdir -p ./logs/${DATASET}
fi

if ! [ -d "${DIR}" ]; then
    mkdir -p ${DIR}
fi

for SEED in $SEEDS
do
    nohup ./scripts/${METHOD}.sh ${DATASET} ${SEED} ${GPU_IDX} ${DEBUG} ${MEM_SIZE} > ${DIR}/seed_${SEED}.out 6>&0 &
done