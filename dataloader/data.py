from datasets import (
    load_dataset, 
    Dataset, 
    DatasetDict, 
    Value,
    ClassLabel,
)
import random
import json
import sys
sys.path.append("..")

chemprot_files = {"train": "../dataloader/chemprot/train.jsonl", 
                "dev": "../dataloader/chemprot/dev.jsonl", 
                "test": "../dataloader/chemprot/test.jsonl"}
bioasq_files = {"train": "../dataloader/bioasq/train.json",  
                "test": "../dataloader/bioasq/test.json"}

CNT_CHEMPROT_ORIGIN = {
    "UPREGULATOR": 0,
    "ACTIVATOR": 0,
    "INDIRECT-UPREGULATOR": 0,
    "DOWNREGULATOR": 0,
    "INHIBITOR": 0,
    "INDIRECT-DOWNREGULATOR": 0,
    "AGONIST": 0,
    "AGONIST-ACTIVATOR": 0,
    "AGONIST-INHIBITOR": 0,
    "ANTAGONIST": 0,
    "SUBSTRATE": 0,
    "PRODUCT-OF": 0,
    "SUBSTRATE_PRODUCT-OF": 0,
}
CNT_CHEMPROT = {"UPREGULATOR": 0, "DOWNREGULATOR": 0, "AGONIST": 0, "ANTAGONIST": 0, "SUBSTRATE": 0}

def format_label(dataset):
    new_features = dataset.features.copy()
    new_features['text'] = Value('string')
    new_features['label'] = ClassLabel(names=list(set(dataset["label"])))
    return dataset.cast(new_features)

def convert_label_chemprot(example):
    label = example["label"]
    CNT_CHEMPROT_ORIGIN[label] += 1
    if label == "UPREGULATOR" or label == "ACTIVATOR" or label == "INDIRECT-UPREGULATOR":
        label = "UPREGULATOR"
    elif label == "DOWNREGULATOR" or label == "INHIBITOR" or label == "INDIRECT-DOWNREGULATOR":
        label = "DOWNREGULATOR"
    elif label == "AGONIST" or label == "AGONIST-ACTIVATOR" or label == "AGONIST-INHIBITOR":
        label = "AGONIST"
    elif label == "ANTAGONIST":
        label = "ANTAGONIST"
    elif label == "SUBSTRATE" or label == "PRODUCT-OF" or label == "SUBSTRATE_PRODUCT-OF":
        label = "SUBSTRATE"
    else:
        print("ERROR: no such label: ", label)

    CNT_CHEMPROT[label] += 1
    example["label"] = label
    return example

def get_dataset_bioasq(data_files, sep_token="<sep>"):
    dataset_dict = {"train": None, "dev": None, "test": None}
    dataset_train = {"text": [], "label": []}
    dataset_dev = {"text": [], "label": []}
    dataset_test = {"text": [], "label": []}
    
    with open(data_files["train"], 'r') as f:
        raw = json.load(f)
        all_len = len(raw)
        train_len = int(all_len * 0.9)
        cnt = 0
        for value in raw:
            question = value["question"]
            for text in value["text"]:
                question = question + sep_token + text
            if cnt < train_len:
                dataset_train["text"].append(question)
                dataset_train["label"].append(value["anwser"])
            else:
                dataset_dev["text"].append(question)
                dataset_dev["label"].append(value["anwser"])
            cnt += 1

        dataset_train = Dataset.from_dict(dataset_train)
        dataset_train = format_label(dataset_train)
        dataset_dev = Dataset.from_dict(dataset_dev)
        dataset_dev = format_label(dataset_dev)
        dataset_dict["train"] = dataset_train
        dataset_dict["dev"] = dataset_dev
    
    with open(data_files["test"], 'r') as f:
        raw = json.load(f)
        for value in raw:
            question = value["question"]
            for text in value["text"]:
                question = question + sep_token + text
            dataset_test["text"].append(question)
            dataset_test["label"].append(value["anwser"])

        dataset_test = Dataset.from_dict(dataset_test)
        dataset_test = format_label(dataset_test)
        dataset_dict["test"] = dataset_test

    dataset = DatasetDict(dataset_dict)
    return dataset

def get_dataset_from_name(dataset_name, sep_token):
    if dataset_name[:8] == "chemprot":
        dataset = load_dataset("json", data_files=chemprot_files)
        dataset = dataset.remove_columns("metadata")
        # dataset["train"] = format_label(dataset["train"]) 
        # dataset["dev"] = format_label(dataset["dev"])
        # dataset["test"] = format_label(dataset["test"])
        dataset["train"] = format_label(dataset["train"].map(convert_label_chemprot))
        print(CNT_CHEMPROT_ORIGIN)
        print(CNT_CHEMPROT)
        dataset["dev"] = format_label(dataset["dev"].map(convert_label_chemprot))
        dataset["test"] = format_label(dataset["test"].map(convert_label_chemprot))
    elif dataset_name[:6] == "bioasq":
        dataset = get_dataset_bioasq(bioasq_files, sep_token)
    
    if dataset_name[-2:] == "fs":
        if dataset["train"].features["label"].num_classes <= 5:
            select_nums = random.sample(range(dataset["train"].num_rows), 32)
            dataset["train"] = dataset["train"].select(select_nums)
        else:
            num_label = dataset["train"].features["label"].num_classes
            names = dataset["train"].features["label"].names
            dataset_train = {"text": [], "label": []}
            for i in range(num_label):
                dset_filter = dataset.filter(lambda x: x["label"] == i)
                dataset_train["text"] += dset_filter["text"]
                dataset_train["label"] += [names[j] for j in dset_filter["label"]]
            dataset_train = Dataset.from_dict(dataset_train)
            dataset["train"] = dataset_train             
    
    return dataset


def get_dataset_from_list(dataset_name, sep_token):
    dataset_dict = {
        "train": {"text": [], "label": []}, 
        "dev": {"text": [], "label": []}, 
        "test": {"text": [], "label": []}
    }
    types = ["train", "dev", "test"]
    index = 0
    for dset_name in dataset_name:
        dset = get_dataset_from_name(dset_name, sep_token)
        for t in types:
            names = dset[t].features["label"].names
            dataset_dict[t]["text"] += dset[t]["text"]
            dataset_dict[t]["label"] += [names[i] for i in dset[t]["label"]]

    for t in types:
        dataset_dict[t] = Dataset.from_dict(dataset_dict[t])
        dataset_dict[t] = format_label(dataset_dict[t])

    dataset = DatasetDict(dataset_dict)
    return dataset    


def get_dataset(dataset_name, sep_token):
    '''
    dataset_name: str, the name of the dataset
    '''
    dataset = None

    # your code for preparing the dataset...
    if isinstance(dataset_name, str):
        dataset = get_dataset_from_name(dataset_name, sep_token)
    elif isinstance(dataset_name, list):
        dataset = get_dataset_from_list(dataset_name, sep_token)

    return dataset