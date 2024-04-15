# Parallelized MNLI
Data Parallelism - DDP by Torch

Model parallelism - Megatron-LM by NVIDIA

Pipeline parallelism - GPipe

Dataset: 
Number of training examples: **392702**;
Number of validation examples: **9823**;
Number of testing examples: **9824**.


Hyperparameters for ```bert-large-cased```:

| Variable | Value |
| --- | --- |
| num_hidden_layers | `6` |
| num_attention_heads | `6` |
| hidden_size | `768` |
| intermediate_size | `2048` |
| max_position_embeddings | `1024` |
| dim_feedforward | `2048` |
| learning_rate | `1e-5` |
| BATCH | `64` |
| N_EPOCHS | `3` |

