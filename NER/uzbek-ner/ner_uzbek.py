# -*- coding: utf-8 -*-
"""NER_uzbek.ipynb

Original file is located at
    https://colab.research.google.com/drive/14e7oDmqJy8PAS25rWg0fToPIxz-mj6jh
"""
import numpy as np
import evaluate
from datasets import load_dataset, Dataset, DatasetDict
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    TrainingArguments,
    Trainer,
    DataCollatorForTokenClassification,
)
from huggingface_hub import HfApi, upload_folder
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model
from sklearn.model_selection import train_test_split
from collections import Counter

# --- Configuration Parameters ---
dataset_name = "risqaliyevds/uzbek_ner"
model_name = "bert-base-multilingual-cased"
peft_output_model_name = "uzbek-ner"
final_output_model_dir = "./ner-uzbek-model"
huggingface_repo_name = "ibodullo2205/uzbek-ner"
training_output_dir = "./ner-uzbek-results" # Corrected from "./ner-uzbek" to avoid conflict
logging_dir = "./logs"
# --- End Configuration Parameters ---

# Load Dataset
dataset = load_dataset(dataset_name)

# Load Pretrained Tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Define base NER entity types
BASE_LABEL_LIST = [
    'CARDINAL', 'DATE', 'EMAIL', 'EVENT', 'FAC', 'FACILITY', 'GPE', 'JCH-2022',
    'JOURNAL', 'LANGUAGE', 'LAW', 'LOC', 'MISC', 'MONEY', 'NORP', 'ORDINAL',
    'ORG', 'PER', 'PERCENT', 'PERIOD', 'PERSON', 'PHONE', 'PRODUCT', 'QUANTITY',
    'RASUM', 'SOCIAL_MEDIA', 'TIME', 'WEBSITE', 'WORK_OF_ART'
]

# Create the full list of NER tags (including O, B-TAG, I-TAG)
# This will be the definitive set of labels for the model.
_final_labels = ["O"]
for label in BASE_LABEL_LIST:
    _final_labels.extend([f"B-{label}", f"I-{label}"])

# These are the final, fixed label mappings to be used throughout the script
final_label2id = {label: i for i, label in enumerate(_final_labels)}
final_id2label = {i: label for i, label in enumerate(_final_labels)}
num_labels = len(_final_labels)

# Keep track of seen unknown ner_categories to avoid repetitive logging
seen_unknown_categories = set()

# --- Data Analysis (Optional - for understanding label distribution) ---
# This would need to be adapted if you want to see counts for B-TAG, I-TAG, O
# from collections import Counter

# ner_tags = [tag for sublist in dataset['train']['ner'] for tag in sublist]
# tag_counts = Counter(ner_tags)
# print("Original tag counts:", tag_counts)
# print("List of all NER tags in training data:", ner_tags) # This can be very verbose
# --- End Data Analysis ---

# --- Train-Test Split ---
# --- Train-Test-Validation Split ---
# The dataset initially has a 'train' split. We need to split it.
if 'train' not in dataset:
    raise ValueError("Original dataset does not contain 'train' split.")

all_data_df = dataset['train'].to_pandas()

# Split into train (70%) and temp (30% for validation + test)
train_df, temp_df = train_test_split(all_data_df, test_size=0.3, random_state=42, shuffle=True)

# Split temp (30%) into validation (15%) and test (15%)
# This means validation and test will each be 0.15 / 0.30 = 50% of temp_df
val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42, shuffle=True)

processed_dataset = DatasetDict({
    'train': Dataset.from_pandas(train_df, preserve_index=False),
    'validation': Dataset.from_pandas(val_df, preserve_index=False),
    'test': Dataset.from_pandas(test_df, preserve_index=False)
})

print("Dataset splits created:")
print(f"Train: {len(processed_dataset['train'])} examples")
print(f"Validation: {len(processed_dataset['validation'])} examples")
print(f"Test: {len(processed_dataset['test'])} examples")


# --- Data Preprocessing ---
def preprocess_data(example):
    text = example["text"]
    ner_dict = example["ner"] # This is a dictionary like {'ORG': ['Microsoft', 'Google'], 'PER': ['John Doe']}

    tokenized_inputs = tokenizer(text, truncation=True, is_split_into_words=False, padding="max_length", max_length=512)
    tokens = tokenizer.convert_ids_to_tokens(tokenized_inputs["input_ids"])
    word_ids = tokenized_inputs.word_ids()

    # Initialize labels with "O" (Outside) for all tokens
    labels_for_tokens = ["O"] * len(tokens) # Renamed from `labels` to avoid confusion

    for ner_category, entities in ner_dict.items():
        upper_ner_category = ner_category.upper() # From the original dataset, keys are like 'CARDINAL', 'DATE' etc.
        
        if upper_ner_category not in BASE_LABEL_LIST:
            if upper_ner_category not in seen_unknown_categories:
                print(f"Warning: Unknown NER category '{upper_ner_category}' found in data. Mapping its tokens to 'O'.")
                seen_unknown_categories.add(upper_ner_category)
            # Tokens for this unknown category will remain "O" as initialized
            continue

        if entities:
            for entity_text in entities:
                entity_tokens = tokenizer.tokenize(entity_text)
                if not entity_tokens:
                    continue
                
                start_idx = -1
                for i in range(len(tokens) - len(entity_tokens) + 1):
                    if tokens[i : i + len(entity_tokens)] == entity_tokens:
                        start_idx = i
                        break
                
                if start_idx != -1:
                    # Assign B-TAG for the first token of the entity
                    # Ensure the generated B-TAG and I-TAG are in our final_label2id
                    b_tag = f"B-{upper_ner_category}"
                    i_tag = f"I-{upper_ner_category}"

                    if b_tag in final_label2id: # Should always be true if BASE_LABEL_LIST is correct
                        labels_for_tokens[start_idx] = b_tag
                        if i_tag in final_label2id: # Should always be true
                            for idx_offset in range(1, len(entity_tokens)):
                                labels_for_tokens[start_idx + idx_offset] = i_tag
                        else: # Should not happen with current setup
                             print(f"Warning: I-TAG '{i_tag}' is unexpectedly not in final_label2id. Defaulting to 'O'.")

                    else: # Should not happen with current setup
                        print(f"Warning: B-TAG '{b_tag}' is unexpectedly not in final_label2id. Defaulting to 'O'.")


    aligned_labels = []
    previous_word_id = None
    for token_idx, word_id in enumerate(word_ids):
        if word_id is None:
            aligned_labels.append(-100)
        elif word_id != previous_word_id:
            current_token_label_str = labels_for_tokens[token_idx]
            # We now use the fixed final_label2id. No dynamic additions.
            # If current_token_label_str (e.g. "B-UNSEEN_FROM_DATA") is not in final_label2id, map to "O".
            # This can happen if an ner_category was valid (in BASE_LABEL_LIST) but somehow the B-/I- version
            # was not correctly formed or if labels_for_tokens[token_idx] was not properly updated.
            # Given the logic above, labels_for_tokens[token_idx] should always be a valid key or "O".
            aligned_labels.append(final_label2id.get(current_token_label_str, final_label2id["O"]))
        else:
            aligned_labels.append(-100) # Mark sub-tokens
        previous_word_id = word_id
    
    tokenized_inputs["labels"] = aligned_labels
    return tokenized_inputs

# Apply preprocessing
tokenized_datasets = processed_dataset.map(
    preprocess_data, 
    batched=False, 
    load_from_cache_file=False # Ensure fresh processing due to logic changes
)

# --- Model Configuration & Loading ---
# num_labels, final_id2label, final_label2id are now defined globally at the top based on BASE_LABEL_LIST
base_model = AutoModelForTokenClassification.from_pretrained(
    model_name,
    num_labels=num_labels, # From the finalized list
    id2label=final_id2label, # From the finalized list
    label2id=final_label2id  # From the finalized list
)

# PEFT Configuration (LoRA)
# No change needed here regarding label consistency as base_model is now correctly initialized.
model_prepared = prepare_model_for_kbit_training(base_model)
lora_config = LoraConfig(
    r=32,
    lora_alpha=64,
    target_modules=["query_proj", "key_proj", "value_proj", "output_proj", "dense", "classifier"], # Ensure these are correct for the model
    lora_dropout=0.05,
    bias="none",
)
peft_model = get_peft_model(model_prepared, lora_config)
peft_model.print_trainable_parameters()

# --- Data Collator ---
data_collator = DataCollatorForTokenClassification(tokenizer)

# --- Evaluation Metric ---
seqeval_metric = evaluate.load("seqeval")

# final_id2label is already defined globally and is fixed.
# It's passed to the model, so it's the correct one to use for decoding predictions.

def compute_metrics(p):
    predictions, gold_labels = p # Renamed 'labels' to 'gold_labels' for clarity
    predictions = np.argmax(predictions, axis=2) # Shape: (batch_size, seq_len, num_labels) -> (batch_size, seq_len)

    # Decode predictions and labels, removing ignored indices (-100)
    # final_id2label (defined globally) is used here.
    true_predictions_decoded = [
        [final_id2label.get(p_val, "O") for (p_val, l_val) in zip(pred_line, label_line) if l_val != -100]
        for pred_line, label_line in zip(predictions, gold_labels)
    ]
    true_labels_decoded = [
        [final_id2label.get(l_val, "O") for (p_val, l_val) in zip(pred_line, label_line) if l_val != -100]
        for pred_line, label_line in zip(predictions, gold_labels)
    ]
    
    results = seqeval_metric.compute(predictions=true_predictions_decoded, references=true_labels_decoded)
    # Return a dictionary of metrics
    return {
        "precision": results["overall_precision"],
        "recall": results["overall_recall"],
        "f1": results["overall_f1"],
        "accuracy": results["overall_accuracy"],
    }

# --- Training Arguments ---
training_args = TrainingArguments(
    output_dir=training_output_dir,
    evaluation_strategy="epoch",     # Evaluate at the end of each epoch
    learning_rate=5e-5,
    per_device_train_batch_size=16,  # Adjust based on GPU memory
    per_device_eval_batch_size=16,   # Adjust based on GPU memory
    num_train_epochs=5,              # Standard number of epochs for fine-tuning
    weight_decay=0.01,
    logging_dir=logging_dir,
    logging_steps=10,                # Log metrics every 10 steps
    save_strategy="epoch",           # Save model checkpoint at the end of each epoch
    load_best_model_at_end=True,     # Load the best model found during training
    metric_for_best_model="f1",      # Use F1 score to determine the best model
    report_to="none",                # Disable reporting to external services (e.g., W&B)
)

# --- Trainer Initialization ---
trainer = Trainer(
    model=peft_model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"], # Use the new validation split for evaluation during training
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

# --- Start Training ---
trainer.train()

# --- Evaluate on Test Set (After Training) ---
print("\n--- Evaluating on Test Set ---")
if "test" in tokenized_datasets:
    test_results = trainer.evaluate(eval_dataset=tokenized_datasets["test"])
    print("Test Set Evaluation Results:")
    for key, value in test_results.items():
        print(f"  {key}: {value}")
else:
    print("No test set found in tokenized_datasets to evaluate.")

# --- Save Model and Tokenizer (PEFT and Final) ---
# Save PEFT model (adapters)
peft_model.save_pretrained(peft_output_model_name)
tokenizer.save_pretrained(peft_output_model_name) # Save tokenizer alongside PEFT model

# Save full model (if needed, or just use the PEFT model)
# To save the merged model (LoRA weights merged with base model):
# merged_model = peft_model.merge_and_unload() # Corrected: use peft_model
# merged_model.save_pretrained(final_output_model_dir)
# tokenizer.save_pretrained(final_output_model_dir)
# For now, let's stick to saving the PEFT model as per original script's likely intent with LoRA

# The original script uses trainer.save_model() for the final model.
# This saves the full model state, including adapters if PEFT is used.
# If you save the peft_model, it saves only the adapters.
# trainer.save_model() will save the entire model including the base and adapters.
# Let's ensure the final model (potentially merged) is saved if that's the goal for `final_output_model_dir`
print(f"\nSaving final model using trainer.save_model() to {final_output_model_dir}")
trainer.save_model(final_output_model_dir)
# The tokenizer is typically saved with save_pretrained, so saving it again here for final_output_model_dir
tokenizer.save_pretrained(final_output_model_dir)


# --- Hugging Face Hub Upload (Optional) ---
# Ensure you are logged in via `huggingface-cli login` if you want to upload.
# The token is removed for security; authentication should be handled by the environment.
# Consider whether to upload the PEFT adapters (peft_output_model_name) 
# or the full model (final_output_model_dir).
# Uploading PEFT adapters is usually preferred for LoRA.
print(f"\nAttempting to upload PEFT model adapters from ./{peft_output_model_name} to Hugging Face Hub...")
upload_folder(
    repo_id=huggingface_repo_name,
    folder_path=peft_output_model_name, # Uploading the PEFT model adapters
    commit_message="Upload PEFT model adapters and tokenizer",
    # token="YOUR_HF_TOKEN" # Removed: Token should be handled by huggingface-cli login or env variables
)

print(f"\nPEFT Model adapters and tokenizer saved to ./{peft_output_model_name}")
print(f"Full model (trainer.save_model) saved to {final_output_model_dir}")
print(f"Files uploaded to Hugging Face Hub at {huggingface_repo_name} (if authenticated).")

