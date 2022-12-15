import argparse
import logging
import datasets
import torch

import sys
sys.path.append("..")
from dataloader.data import get_dataset

import wandb
import evaluate
import transformers
import numpy as np
from sklearn.metrics import f1_score, accuracy_score
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    BertTokenizer,
    DataCollatorWithPadding,
    EvalPrediction,
    PretrainedConfig,
    Trainer,
    TrainingArguments,
    set_seed,
)
from adapters.adapters import load_adapter_model

wandb.login()

def parse_args():
    parser = argparse.ArgumentParser(description="Finetune a transformers model on a text classification task")
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default="./post_train",
    )
    parser.add_argument(
        "--vocab_path",
        type=str,
        default="./post_train",
    )
    '''
        chemprot: ../dataloader/chemprot/post_train10000-merged-vocabulary_optimized_frt
        bioasq: ../dataloader/bioasq/post_train10000-merged-vocabulary_optimized_frt
    '''
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="chemprot_sup",
    )
    parser.add_argument(
        '--add_adapter',
        default=False,
        help='whether to add adapter'
    )
    parser.add_argument(
        '--do_train',
        default=True,
        help='whether to do training'
    )
    parser.add_argument(
        '--do_eval',
        default=True,
        help='whether to do evaluation'
    )
    parser.add_argument(
        "--num_train_epochs", 
        type=int, 
        default=16, 
        help="Training epochs."
    )
    parser.add_argument(
        "--seed", 
        type=int, 
        default=2022, 
        help="A seed for reproducible training."
    )
    args = parser.parse_args()

    return args

# Get arguments
args = parse_args()

# Set seed before initializing model.
set_seed(args.seed)

logging.basicConfig(filename='./test.log', level=logging.DEBUG)
device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)

# Load tokenizer
print(args.model_name_or_path)
tokenizer = AutoTokenizer.from_pretrained(args.vocab_path)

# Get the datasets
raw_datasets = get_dataset(args.dataset_name, tokenizer.sep_token)

# Labels
label_list = raw_datasets["train"].features["label"].names
num_labels = len(label_list)
print(label_list)
print("num_labels: ", num_labels)

def preprocess_function(examples):
    # Tokenize the texts
    return tokenizer(examples["text"], padding="max_length", max_length=512, truncation=True)

train_dataset = raw_datasets["train"].map(preprocess_function, batched=True)
dev_dataset = raw_datasets["dev"].map(preprocess_function, batched=True)
test_dataset = raw_datasets["test"].map(preprocess_function, batched=True)

# Load pretrained model
config = AutoConfig.from_pretrained(args.model_name_or_path, num_labels=num_labels) 
model = AutoModelForSequenceClassification.from_pretrained(
    args.model_name_or_path,
    config=config,
)
if args.add_adapter:
    model = load_adapter_model(model)

model.cuda()

training_args = TrainingArguments(
    output_dir='../results', 
    report_to="wandb",  
    run_name=args.model_name_or_path+'_'+args.dataset_name+'_'+str(args.seed), 
    evaluation_strategy="epoch",
    logging_steps=100,
)
training_args.seed = args.seed
training_args.num_train_epochs = args.num_train_epochs
training_args.per_device_train_batch_size = 32
training_args.per_eval_train_batch_size = 32

# Get the metric function
# metric = evaluate.load("accuracy")

# You can define your custom compute_metrics function. It takes an `EvalPrediction` object (a namedtuple with a
# predictions and label_ids field) and has to return a dictionary string to float.
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis = 1)
    acc={}
    acc['accuracy'] = accuracy_score(labels, predictions)
    acc['micro-f1'] = f1_score(labels, predictions, average='micro')
    return acc

# Data collator will default to DataCollatorWithPadding when the tokenizer is passed to Trainer, so we change it if
# we already did the padding.
data_collator = DataCollatorWithPadding(tokenizer, padding="max_length", max_length=12)

# Initialize our Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=dev_dataset,
    compute_metrics=compute_metrics,
    tokenizer=tokenizer,
    data_collator=data_collator,
)

if args.do_train:
    train_result = trainer.train()
    metrics = train_result.metrics
    trainer.save_model('../results')  # Saves the tokenizer too for easy upload
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()

if args.do_eval:
    metrics = trainer.evaluate(eval_dataset=test_dataset)
    trainer.log_metrics("eval", metrics)
    trainer.save_metrics("eval", metrics)

wandb.finish()