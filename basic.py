import torch
import torch.nn as nn
from torch import optim
import time
import math
import os

from datasets import load_dataset #, load_metric('glue', 'mnli')
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoConfig
from sklearn.metrics import accuracy_score
from copy import deepcopy
import logging
logger = logging.getLogger(__name__) # DEBUG, info, warning, error
logging.getLogger().setLevel(logging.INFO)
logging.root.setLevel(logging.INFO)
logging.basicConfig(level=logging.INFO)

is_wandb = True

if is_wandb:
    import wandb
    wandb.init(project="paral-mnli")



def tokenize(text):
    return tokenizer(text['premise'], text['hypothesis'], truncation=True)


def train_func(model, train_loader, optimizer, criterion):
    model.train()
    epoch_loss = 0
    se = []
    total_samples = 0
    start_time = time.time()
    
    for batch in train_loader:
        inputs = tokenizer(batch['premise'], batch['hypothesis'], padding=True, truncation=True, return_tensors='pt')
        inputs = {key: inputs[key].to(device) for key in inputs}
        labels = batch['label'].to(device)

        optimizer.zero_grad()
        outputs = model(**inputs)
        loss = criterion(outputs.logits, labels)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
        
        optimizer.step()
        # epoch_loss += loss.item()
        se.append(loss.item())
        total_samples += len(labels)

    epoch_loss = sum(se)
    end_time = time.time()
    elapsed_time = end_time - start_time
    throughput = total_samples / elapsed_time
    se = torch.std(torch.tensor(se)) / math.sqrt(len(train_loader))
    se = se.item()
    
    return epoch_loss / len(train_loader), throughput, se


def evaluate(model, loader, criterion):
    model.eval()
    epoch_loss = []
    accuracy = []
    
    with torch.no_grad():
        for batch in loader:
            inputs = tokenizer(batch['premise'], batch['hypothesis'], padding=True, truncation=True, return_tensors='pt')
            inputs = {key: inputs[key].to(device) for key in inputs}
            labels = batch['label'].to(device)

            outputs = model(**inputs)
            loss = criterion(outputs.logits, labels)
            epoch_loss.append(loss.item())

            accuracy.append(accuracy_score(outputs.logits.argmax(axis=1).cpu().numpy(), labels.cpu().numpy()))
    
    losses = sum(epoch_loss)
    accur = sum(accuracy)/len(val_loader)

    return losses / len(val_loader) , accur


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

# set the pseudo-random generator
manual_seed = 77
torch.manual_seed(manual_seed)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
n_gpu = torch.cuda.device_count()
if n_gpu > 0:
    torch.cuda.manual_seed(manual_seed)


data_path = "LysandreJik/glue-mnli-train"
logger.info(f"\n### Data {data_path} is loading ...\n")
dataset_train = load_dataset(data_path, split='train', trust_remote_code=True)
dataset_val = load_dataset(data_path, split='validation', trust_remote_code=True)
data_val = dataset_val.train_test_split(test_size=0.5)

logger.info(f"Number of training examples: {len(dataset_train)}")
logger.info(f"Number of validation examples: {len(data_val['train'])}")
logger.info(f"Number of testing examples: {len(data_val['test'])}")  

model_name = 'bert-large-cased'
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
#tokenized_datasets = datasets.map(tokenize, batched=True)


train_loader = DataLoader(dataset_train, batch_size=BATCH, shuffle=True) # shuffle ?
val_loader = DataLoader(data_val['train'], batch_size=BATCH, shuffle=False)
test_loader = DataLoader(data_val['test'], batch_size=BATCH, shuffle=False)

print("len(train_loader) {}".format(len(train_loader)))

logger.info(f"\n### Loading {model_name} model ...\n")
default_config = AutoConfig.from_pretrained(model_name, num_labels=num_labels)
modified_config = deepcopy(default_config)
# hidden_size = 768
modified_config.num_hidden_layers = num_hidden_layers
modified_config.max_position_embeddings = max_position_embeddings
modified_config.num_attention_heads = num_attention_heads
modified_config.intermediate_size = intermediate_size
modified_config.hidden_size = hidden_size
model = AutoModelForSequenceClassification.from_config(modified_config).to(device)


# we will ignore the pad token in true target set
TRG_PAD_IDX = tokenizer.pad_token_id

optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
criterion = nn.CrossEntropyLoss(ignore_index = TRG_PAD_IDX)

total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

logger.info(f'\n### Training parameters: ###\n')
logger.info(f"num_hidden_layers {num_hidden_layers}")
logger.info(f"num_attention_heads {num_attention_heads}")
logger.info(f"intermediate_size {intermediate_size}")
logger.info(f"max_position_embeddings {max_position_embeddings}")
logger.info(f"hidden_size {hidden_size}")
logger.info(f"learning_rate {learning_rate}")
logger.info(f"BATCH {BATCH}")
logger.info(f"N_EPOCHS {N_EPOCHS}")
logger.info(f"TRG_PAD_IDX {TRG_PAD_IDX}")
logger.info(f"# Number of trainable parameters: {total_params}")

if is_wandb:
    wandb.run.name = "basic_"+str(BATCH)+"b_"+str(N_EPOCHS)+"ep"


logger.info(f"### Start training...\n")
for epoch in range(N_EPOCHS):
    start_time = time.time()

    train_loss, throughput, se = train_func(model, train_loader, optimizer, criterion)
    valid_loss, accuracy = evaluate(model, val_loader, criterion)
    end_time = time.time()
    epoch_secs = end_time - start_time


    if is_wandb: wandb.log({"valid_accuracy": accuracy, "train_loss": train_loss, "valid_loss": valid_loss, 'epoch_secs':epoch_secs, "throughput": throughput, "se": se})
    logger.info(f'# Epoch: [{epoch+1}/{N_EPOCHS}] | Time: {epoch_secs:.3f}s')
    logger.info(f'\t Train Loss: {train_loss:.3f} | Throughput: {throughput:7.3f}')
    logger.info(f'\t Val.  Loss: {valid_loss:.3f} |         SE: {se:7.3f}')
    logger.info(f'\t Val. Accuracy: {accuracy:.3f}')


logger.info(f"## Start testing...\n")
test_loss, accuracy = evaluate(model, test_loader, criterion)
logger.info(f'\t Test Loss: {test_loss:.3f}\t Test Accuracy: {accuracy:.3f}')
if is_wandb:
    wandb.log({"test_accuracy": accuracy, "test_loss": test_loss})
    wandb.finish()
logger.info("\nTon modèle est très bon!")
logger.info("C'est la fin -_0")


logger.info('I am out ^-^')

