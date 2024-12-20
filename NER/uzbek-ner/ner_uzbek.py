# -*- coding: utf-8 -*-
"""NER_uzbek.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/14e7oDmqJy8PAS25rWg0fToPIxz-mj6jh
"""

!pip install seqeval

!pip install datasets huggingface_hub transformers evaluate

from huggingface_hub import notebook_login

notebook_login()

from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForTokenClassification, TrainingArguments, Trainer
from transformers import DataCollatorForTokenClassification
import evaluate
import numpy as np

# Load Dataset
from datasets import Dataset, DatasetDict
dataset = DatasetDict()
dataset_name = "risqaliyevds/uzbek_ner"
dataset = load_dataset(dataset_name)

# Load Pretrained Tokenizer and Model
model_name = "dbmdz/bert-base-turkish-cased"
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Define NER labels (must match unique keys in the "ner" dictionary)
LABEL_LIST = ['CARDINAL', 'DATE', 'EMAIL', 'EVENT', 'FAC', 'FACILITY', 'GPE', 'JCH-2022', 'JOURNAL', 'LANGUAGE', 'LAW', 'LOC', 'MISC', 'MONEY', 'NORP', 'ORDINAL', 'ORG', 'PER', 'PERCENT', 'PERIOD', 'PERSON', 'PHONE', 'PRODUCT', 'QUANTITY', 'RASUM', 'SOCIAL_MEDIA', 'TIME', 'WEBSITE', 'WORK_OF_ART']
label2id = {label: idx for idx, label in enumerate(LABEL_LIST)}
id2label = {idx: label for label, idx in label2id.items()}

from collections import Counter

ner_tags = [tag for sublist in dataset['train']['ner'] for tag in sublist]
tag_counts = Counter(ner_tags)

print(tag_counts)
print(ner_tags)

from sklearn.model_selection import train_test_split

# Convert the train dataset to a pandas DataFrame
train_df = dataset['train'].to_pandas()

# Split the DataFrame into train and test sets
train_df, test_df = train_test_split(train_df, test_size=0.2, random_state=42)

# Convert the DataFrames back to datasets
dataset['train'] = Dataset.from_pandas(train_df)
dataset['test'] = Dataset.from_pandas(test_df)

def preprocess_data(example):
    text = example["text"]
    ner_dict = example["ner"]

    tokenized_inputs = tokenizer(text, truncation=True, is_split_into_words=False)
    tokens = tokenizer.convert_ids_to_tokens(tokenized_inputs["input_ids"]) 
    word_ids = tokenized_inputs.word_ids() 
    labels = ["O"] * len(tokens)  

    for key, values in ner_dict.items():
        if values:  
            for entity in values:
                entity_tokens = tokenizer.tokenize(entity) 

                start_idx = -1
                for i in range(len(tokens) - len(entity_tokens) + 1): 
                    if tokens[i : i + len(entity_tokens)] == entity_tokens:
                        start_idx = i
                        break

                if start_idx != -1: 
                    for idx in range(len(entity_tokens)): 
                        if idx == 0: 
                            labels[start_idx + idx] = f"B-{key.upper()}"
                        else:
                            labels[start_idx + idx] = f"I-{key.upper()}"

    aligned_labels = []
    for word_idx in word_ids:
        if word_idx is None:
            aligned_labels.append(-100)
        else:
            if labels[word_idx] not in label2id:
                new_label_id = len(label2id)
                label2id[labels[word_idx]] = new_label_id
                id2label[new_label_id] = labels[word_idx]

            aligned_labels.append(label2id[labels[word_idx]]) 

    tokenized_inputs["labels"] = aligned_labels
    return tokenized_inputs

tokenized_datasets = dataset.map(preprocess_data, batched=False)

num_labels = len(label2id) 
model = AutoModelForTokenClassification.from_pretrained(
    model_name, num_labels=num_labels, id2label=id2label, label2id=label2id
)

from peft import prepare_model_for_kbit_training
from peft import LoraConfig, PeftModel, LoraModel, get_peft_model

model = prepare_model_for_kbit_training(model)

config = LoraConfig(
    r=32,
    lora_alpha=64,
    target_modules=["q_proj", "k_proj", "v_proj", "out_proj", "fc1", "fc2"],
    lora_dropout=0.05,
    bias="none",
)

model = get_peft_model(model, config)

model.print_trainable_parameters()
# Data Collator
data_collator = DataCollatorForTokenClassification(tokenizer)

# Evaluation Metric
metric = evaluate.load("seqeval")

def compute_metrics(pred):
    predictions, labels = pred
    predictions = np.argmax(predictions, axis=2)

    true_predictions = [
        [id2label[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [id2label[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    return metric.compute(predictions=true_predictions, references=true_labels)

# Training Arguments
training_args = TrainingArguments(
    output_dir="./ner-uzbek",
    evaluation_strategy="epoch",
    learning_rate=5e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_dir="./logs",
    save_strategy="epoch", 
    load_best_model_at_end=True,
    metric_for_best_model="overall_f1", 
    report_to="none",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["test"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

trainer.train()

from huggingface_hub import HfApi
model.save_pretrained("uzbek-ner")
tokenizer.save_pretrained("uzbek-ner")
repo_name = "ibodullo2205/uzbek-ner"

from huggingface_hub import upload_folder

upload_folder(
    repo_id=repo_name,
    folder_path="./uzbek-ner", 
    commit_message="Upload my model from Google Colab",
    token="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"
)

trainer.save_model("./ner-uzbek-model")

