#!/bin/bash

# CIL CONFIG
N=100
M=0
GPU_TRANSFORM="--gpu_transform"
USE_AMP="--use_amp"
NUM_GPUS=1
ROOT="/data/icml2023_hierCL/data"

NOTE=$1 # Short description of the experiment. (WARNING: logs/results with the same note will be overwritten!)
if [ "$NOTE" == "ewc" ]; then
    MODE="ewc++"
else
    MODE=$NOTE
fi

DATASET=$2 # cifar10, cifar100, tinyimagenet, imagenet, cifar100_super, imagenet_subset, imagenet_subset_shuffle
SEEDS=$3 #"0" #"1 2 3"
GPU_IDX=$4 #  0, 1, 2, 3 "1"
DEBUG=$5 #"False"
MEM_SIZE=$6
WORKERS_PER_GPU=$7
TEMP=$8
#echo $TEMP


if [ "$DATASET" == "cifar10" ]; then
    N_TASKS=10 MEM_SIZE=500 ONLINE_ITER=1
    MODEL_NAME="resnet18" EVAL_PERIOD=100
    BATCHSIZE=16; LR=3e-4 OPT_NAME="adam" SCHED_NAME="default" MEMORY_EPOCH=100

elif [ "$DATASET" == "cifar100" ]; then
    N_TASKS=5 MEM_SIZE=2000 ONLINE_ITER=3
    N=50 M=10
    MODEL_NAME="resnet34" EVAL_PERIOD=500
    BATCHSIZE=16; LR=5e-2 OPT_NAME="sgd" SCHED_NAME="cos" MEMORY_EPOCH=256

elif [ "$DATASET" == "cifar100_super" ]; then
    N_TASKS=10 MEM_SIZE=2000 ONLINE_ITER=3
    MODEL_NAME="resnet34" EVAL_PERIOD=100
    BATCHSIZE=16; LR=3e-4 OPT_NAME="adam" SCHED_NAME="default" MEMORY_EPOCH=100
    
elif [ "$DATASET" == "tinyimagenet" ]; then
    N_TASKS=10 MEM_SIZE=4000 ONLINE_ITER=3
    MODEL_NAME="resnet34" EVAL_PERIOD=100 
    BATCHSIZE=32; LR=3e-4 OPT_NAME="adam" SCHED_NAME="default" MEMORY_EPOCH=100

elif [ "$DATASET" == "imagenet" ]; then
    N_TASKS=10 MEM_SIZE=20000 ONLINE_ITER=0.25
    MODEL_NAME="resnet34" EVAL_PERIOD=1000
    BATCHSIZE=256; LR=3e-4 OPT_NAME="adam" SCHED_NAME="default" MEMORY_EPOCH=100
 


elif [ "$DATASET" == "cifar100_hier_setup1" ]; then
    N_TASKS=6 MEM_SIZE=2000 ONLINE_ITER=3
    MODEL_NAME="resnet34" EVAL_PERIOD=1000
    BATCHSIZE=16; LR=3e-4 OPT_NAME="adam" SCHED_NAME="default" MEMORY_EPOCH=100
    
elif [ "$DATASET" == "cifar100_hier_setup2" ]; then
    N_TASKS=6 MEM_SIZE=2000 ONLINE_ITER=3
    MODEL_NAME="resnet34" EVAL_PERIOD=1000
    BATCHSIZE=16; LR=3e-4 OPT_NAME="adam" SCHED_NAME="default" MEMORY_EPOCH=100
    
elif [ "$DATASET" == "stanford_car_setup1" ]; then
    N_TASKS=10 MEM_SIZE=1000 ONLINE_ITER=0.5
    MODEL_NAME="resnet34" EVAL_PERIOD=200
    BATCHSIZE=64; LR=3e-4 OPT_NAME="adam" SCHED_NAME="default" MEMORY_EPOCH=100
    
elif [ "$DATASET" == "stanford_car_setup2" ]; then
    N_TASKS=10 MEM_SIZE=1000 ONLINE_ITER=0.5
    MODEL_NAME="resnet34" EVAL_PERIOD=200
    BATCHSIZE=64; LR=3e-4 OPT_NAME="adam" SCHED_NAME="default" MEMORY_EPOCH=100   
    
elif [ "$DATASET" == "imagenet_subset" ]; then
    N_TASKS=6 MEM_SIZE=5000 ONLINE_ITER=0.25
    MODEL_NAME="resnet34" EVAL_PERIOD=2500
    BATCHSIZE=64; LR=3e-4 OPT_NAME="adam" SCHED_NAME="default" MEMORY_EPOCH=100   
    
elif [ "$DATASET" == "imagenet_subset_setup2" ]; then
    N_TASKS=6 MEM_SIZE=5000 ONLINE_ITER=0.25
    MODEL_NAME="resnet34" EVAL_PERIOD=2500
    BATCHSIZE=64; LR=3e-4 OPT_NAME="adam" SCHED_NAME="default" MEMORY_EPOCH=100   
    
elif [ "$DATASET" == "imagenet_subset_sub_shuffle" ]; then
    N_TASKS=6 MEM_SIZE=5000 ONLINE_ITER=0.25
    MODEL_NAME="resnet34" EVAL_PERIOD=2500
    BATCHSIZE=64; LR=3e-4 OPT_NAME="adam" SCHED_NAME="default" MEMORY_EPOCH=100 
    
else
    echo "Undefined setting"
    exit 1
fi

for RND_SEED in $SEEDS
do
    python main.py --mode $MODE \
    --dataset $DATASET \
    --n_tasks $N_TASKS --m $M --n $N \
    --rnd_seed $RND_SEED \
    --model_name $MODEL_NAME --opt_name $OPT_NAME --sched_name $SCHED_NAME \
    --lr $LR --batchsize $BATCHSIZE \
    --memory_size $MEM_SIZE $GPU_TRANSFORM --online_iter $ONLINE_ITER \
    --note $NOTE --eval_period $EVAL_PERIOD $USE_AMP --gpu_idx $GPU_IDX --num_gpus $NUM_GPUS --workers_per_gpu $WORKERS_PER_GPU --memory_epoch $MEMORY_EPOCH --debug $DEBUG --root $ROOT --temp $TEMP
done
