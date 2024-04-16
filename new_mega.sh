pip install wandb
wandb login "key"
pip install --upgrade pip
rm -r Megatron-LM
git clone https://github.com/profii/Megatron-LM
cd Megatron-LM
pip install ./
# pip list
cd data
unzip train_cut.zip
cd ../tasks
mkdir checkpoints
cd checkpoints
mkdir bert_345m
cd bert_345m
wget --content-disposition https://api.ngc.nvidia.com/v2/models/nvidia/megatron_bert_345m/versions/v0.1_cased/zip -O megatron_bert_345m_v0.1_cased.zip
unzip megatron_bert_345m_v0.1_cased.zip
cd ../../..

WORLD_SIZE=2
BATCH=64
MICRO_BATCH=64

DISTRIBUTED_ARGS="--nproc_per_node $WORLD_SIZE \
                  --nnodes 1 \
                  --node_rank 0 \
                  --master_addr localhost \
                  --master_port 6000"
LOGGING_ARGS="
        --wandb-project ${WANDB_PROJECT:-"paral-mnli"}
        --wandb-exp-name ${WANDB_NAME:-"megatron_64b_3ep"}"

TRAIN_DATA="data/train_cut.tsv"
VALID_DATA="data/dev.tsv"
PRETRAINED_CHECKPOINT="/data/Megatron-LM/tasks/checkpoints/bert_345m"
VOCAB_FILE="data/vocab.txt"
CHECKPOINT_PATH="/data/Megatron-LM/tasks/checkpoints/bert_mnli"
# CUDA_DEVICE_MAX_CONNECTIONS=1 python -m torch.distributed.launch
# CUDA_DEVICE_MAX_CONNECTIONS=1 torchrun --wandb-project="paral_mnli"
CUDA_DEVICE_MAX_CONNECTIONS=1 python -m torch.distributed.launch $DISTRIBUTED_ARGS /data/Megatron-LM/tasks/main.py \
               --task MNLI \
               --seed 77 \
               --train-data $TRAIN_DATA \
               --valid-data $VALID_DATA \
               --tokenizer-type BertWordPieceCase \
               --vocab-file $VOCAB_FILE \
               --epochs 3 \
               --pretrained-checkpoint-dir $PRETRAINED_CHECKPOINT \
               --tensor-model-parallel-size $WORLD_SIZE \
               --pipeline-model-parallel-size 1 \
               --num-layers 6 \
               --hidden-size 768 \
               --num-attention-heads 6 \
               --micro-batch-size $MICRO_BATCH \
               --lr 1.0e-5 \
               --weight-decay 1.0e-2 \
               --lr-decay-style linear \
               --lr-warmup-fraction 0.065 \
               --seq-length 256 \
               --max-position-embeddings 1024 \
               --save-interval 500000 \
               --save $CHECKPOINT_PATH \
               --log-interval 1000 \
               --eval-interval 5953 \
               --eval-iters 50 \
               --fp16 \
               $LOGGING_ARGS
