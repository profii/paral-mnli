# $ torchrun --nproc-per-node 4 pippy_bert.py

import argparse
import os

import torch
import torch.nn as nn
import torch.distributed as dist
import time
from pippy import pipeline
from pippy.PipelineSchedule import ScheduleGPipe
from pippy.PipelineStage import PipelineStage
from pippy.IR import SplitPoint, MultiUseParameterConfig, Pipe, PipeSplitWrapper, annotate_split_points
from pippy.PipelineDriver import PipelineDriverFillDrain, PipelineDriver1F1B, PipelineDriverInterleaved1F1B, PipelineDriverBase
from copy import deepcopy
from datasets import load_dataset
from torch.utils.data import DataLoader

from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoConfig
from transformers import BertForSequenceClassification
from hf_utils import generate_inputs_for_model, get_number_of_params


def add_split_points(bert, encoders_per_rank):
    for i in range(0, bert.config.num_hidden_layers // encoders_per_rank):
        annotate_split_points(bert,
                              {f'bert.encoder.layer.{i * encoders_per_rank}': PipeSplitWrapper.SplitPoint.BEGINNING})
    annotate_split_points(bert, {f'cls': PipeSplitWrapper.SplitPoint.BEGINNING})
    return bert.config.num_hidden_layers // encoders_per_rank + 2


def run(args):
    # Model configs
    # Model configs
    manual_seed = 77
    num_labels = 3
    num_hidden_layers = 6 # 12
    num_attention_heads = 6 # 12  # should % n_head == 0
    hidden_size = 768 # 1024
    intermediate_size = 2048 #3072
    max_position_embeddings = 1024
    # hidden_dropout_prob = 0.1 # dropout = 0.1
    learning_rate = 1e-5
    BATCH = 64 # 64
    N_EPOCHS = 3 # 3
    model_name = 'bert-large-cased'
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)

    default_config = AutoConfig.from_pretrained(model_name, num_labels=num_labels)
    modified_config = deepcopy(default_config)
    # hidden_size = 768
    modified_config.num_hidden_layers = num_hidden_layers
    modified_config.max_position_embeddings = max_position_embeddings
    modified_config.num_attention_heads = num_attention_heads
    modified_config.intermediate_size = intermediate_size
    modified_config.hidden_size = hidden_size
    bert = AutoModelForSequenceClassification.from_config(modified_config)
    # config = BertConfig()
    print("Using device:", args.device)
    # Create model
    # model_class = BertForMaskedLM
    # model_name = "BertForMaskedLM"
    # bert = model_class(config)
    model_name = 'BertForSequenceClassification'
    model_class = BertForSequenceClassification
    bert.to(args.device)
    if args.rank == 0:
        print(bert.config)
        print(f"Total number of params = {get_number_of_params(bert) // 10 ** 6}M")

    

    data_path = "LysandreJik/glue-mnli-train"
    dataset_train = load_dataset(data_path, split='train', trust_remote_code=True)
    dataset_val = load_dataset(data_path, split='validation', trust_remote_code=True)
    # #Input configs
    # example_inputs = generate_inputs_for_model(
    #     model_class, bert, model_name, args.batch_size, args.device)
    
    # if args.rank == 0:
        # print(example_inputs['input_ids'].shape)
    # Annotate split points
    # add_split_points(bert, args.world_size)
    emb_head = 2  
    master_emb_head = 1 + emb_head  
    encoders_per_rank = (bert.config.num_hidden_layers + (args.world_size - master_emb_head) - 1) // (
            args.world_size - master_emb_head)  
    print(f"encoders_per_rank = {encoders_per_rank}")
    number_of_workers = emb_head + bert.config.num_hidden_layers // encoders_per_rank
    print(f"number_of_workers = {number_of_workers}")

    device = args.device
    print("Using device:", device)

    sm_cnt = add_split_points(bert, encoders_per_rank)

    # Create pipeline
    bert_pipe = pipeline(
        bert,
        num_chunks=args.chunks,
        example_args=()
    )
    
    assert bert_pipe.num_stages == args.world_size, f"nstages = {bert_pipe.num_stages} nranks = {args.world_size}"
    if args.rank == 0:
        print(bert_pipe)
        for i, sm in enumerate(bert_pipe.split_gm.children()):
            print(f"Pipeline stage {i} {get_number_of_params(sm) // 10 ** 6}M params")

    # Create schedule runtime
    stage = PipelineStage(
        bert_pipe,
        args.rank,
        device=args.device,
    )
    driver = PipelineDriverFillDrain(
        bert_pipe,
        64,
        world_size=args.world_size,
    )
    train(bert_pipe, stage, dataset_train, driver, args, tokenizer)

def train(model, stage, data, driver, args, tokenizer): 
    loss_fn = nn.CrossEntropyLoss()
    schedule = ScheduleGPipe(stage, args.chunks, loss_fn=loss_fn)
    optimizer = driver.instantiate_optimizer(torch.optim.Adam)
    data = data.train_test_split(test_size=0.2)
    train_loader = DataLoader(data['train'], batch_size=64, shuffle=False)
    for epoch in range(3):
        start_time = time.time()
        model.train()
        loss = []
        for batch in train_loader:
            inputs = tokenizer(batch['premise'], batch['hypothesis'], padding=True, truncation=True, return_tensors='pt')
            if args.rank == 0:
                schedule.step(**inputs)
            elif args.rank == args.world_size - 1:
                out = schedule.step()
                optimizer.zero_grad()
                pipe_loss = driver(out, batch['label'])
                optimizer.step()
                loss.append(pipe_loss.item())
                total_samples += len(batch['label'])
                print("loss: {pipe_loss}}")
            else:
                schedule.step()
        epoch_time = time.time() - start_time
        print(f"Epoch {epoch + 1}: {epoch_time} seconds")
        eval_accuracy, eval_loss = eval(model, schedule, data['test'], driver, args, tokenizer)
        print(f"Accuracy: {eval_accuracy}; Loss: {eval_loss}")

def eval(model, schedule, data, driver, args, tokenizer):
    model.eval()
    total = 0
    true_total = 0
    val_loader = DataLoader(data, batch_size=64, shuffle=False)
    loss = 0.0
    for batch in val_loader:
        inputs = tokenizer(batch['premise'], batch['hypothesis'], padding=True, truncation=True, return_tensors='pt')
        if args.rank == 0:
            schedule.step(**inputs)
        elif args.rank == args.world_size - 1:
            out = schedule.step()
            pipe_loss = driver(out, batch['label'])
            loss += pipe_loss
            predictions = torch.argmax(out, dim=-1)
            true_pred = (predictions == batch['label'])
            true_total += true_pred
            total += len(batch['label'])
        else:
            schedule.step()
    return (true_total / total, loss / total)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--world_size', type=int, default=int(os.getenv("WORLD_SIZE", 4)))
    parser.add_argument('--rank', type=int, default=int(os.getenv("RANK", -1)))
    parser.add_argument('--master_addr', type=str, default=os.getenv('MASTER_ADDR', 'localhost'))
    parser.add_argument('--master_port', type=str, default=os.getenv('MASTER_PORT', '29500'))
    parser.add_argument('--schedule', type=str, default="FillDrain")
    parser.add_argument('--cuda', type=int, default=int(torch.cuda.is_available()))
    parser.add_argument("--chunks", type=int, default=4)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--batches', type=int, default=1)

    args = parser.parse_args()

    if args.cuda:
        dev_id = args.rank % torch.cuda.device_count()
        args.device = torch.device(f"cuda:{dev_id}")
    else:
        args.device = torch.device("cpu")

    # Init process group
    backend = "nccl" if args.cuda else "gloo"
    dist.init_process_group(
        backend=backend,
        rank=args.rank,
        world_size=args.world_size,
    )

    run(args)
