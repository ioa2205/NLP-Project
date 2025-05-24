# Uzbek NER Model

This project focuses on Named Entity Recognition (NER) for the Uzbek language. It utilizes a Transformer-based model fine-tuned on a publicly available Uzbek NER dataset.

## Model Architecture

The model uses a Transformer base, currently configured as `bert-base-multilingual-cased`. To enable efficient fine-tuning, Low-Rank Adaptation (LoRA) is employed.

The main training script for this model is located at `NER/uzbek-ner/ner_uzbek.py`.

## Dataset

The project uses the `risqaliyevds/uzbek_ner` dataset available on the Hugging Face Hub. Since this dataset originally provides only a single 'train' split, the `NER/uzbek-ner/ner_uzbek.py` script is responsible for splitting this data into:
*   Training set: 70%
*   Validation set: 15%
*   Test set: 15%

This allows for proper model training, validation during training, and final evaluation on an unseen test set.

## Label Set

The NER model identifies entities based on the following base categories (defined as `BASE_LABEL_LIST` in the training script):
`CARDINAL`, `DATE`, `EMAIL`, `EVENT`, `FAC`, `FACILITY`, `GPE`, `JCH-2022`, `JOURNAL`, `LANGUAGE`, `LAW`, `LOC`, `MISC`, `MONEY`, `NORP`, `ORDINAL`, `ORG`, `PER`, `PERCENT`, `PERIOD`, `PERSON`, `PHONE`, `PRODUCT`, `QUANTITY`, `RASUM`, `SOCIAL_MEDIA`, `TIME`, `WEBSITE`, `WORK_OF_ART`.

These base tags are used with the standard IOB tagging scheme:
*   **B-TAG**: Beginning of an entity (e.g., B-PER for the first token of a Person entity).
*   **I-TAG**: Inside of an entity (e.g., I-PER for subsequent tokens of a Person entity).
*   **O**: Outside of any named entity.

The full list of labels used by the model includes "O" and "B-TYPE", "I-TYPE" for each base category (e.g., "B-PER", "I-PER", "B-LOC", "I-LOC", etc.).

## Setup and Training

1.  **Dependencies**: All required Python packages are listed in `NER/requirements.txt`. They can be installed using pip:
    ```bash
    pip install -r NER/requirements.txt
    ```

2.  **Training**: The model can be trained by directly executing the main script:
    ```bash
    python NER/uzbek-ner/ner_uzbek.py
    ```

3.  **Hugging Face Hub Authentication (Optional)**: If you intend to upload the fine-tuned model (or adapters) to the Hugging Face Hub, you need to authenticate first using the Hugging Face CLI:
    ```bash
    huggingface-cli login
    ```
    The script includes a feature to upload the PEFT model adapters.

## Evaluation

The training script evaluates the model's performance using the `seqeval` library, reporting metrics such as precision, recall, and F1-score for entity recognition.
*   Evaluation is performed on the **validation set** at the end of each training epoch. The best model based on F1-score on the validation set is saved.
*   After training is complete, a final evaluation is performed on the **test set**.

**Note**: Due to current environment limitations during the automated refactoring process, the model was not re-trained. The results from a new training run should be manually added here after executing the script.

## Configuration

Key parameters for training and model selection can be configured by modifying the variables defined at the beginning of the `NER/uzbek-ner/ner_uzbek.py` script. These include:
*   `dataset_name`: The Hugging Face dataset identifier.
*   `model_name`: The base Transformer model to use.
*   `peft_output_model_name`: Directory name for saving PEFT adapters.
*   `final_output_model_dir`: Directory name for saving the full model using `trainer.save_model()`.
*   `huggingface_repo_name`: Target repository ID on Hugging Face Hub for uploads.
*   `training_output_dir`: Directory for saving training outputs (checkpoints, logs).
*   `logging_dir`: Directory for TensorBoard logs.
*   Hyperparameters like `num_train_epochs`, `learning_rate`, batch sizes, etc., are configured within the `TrainingArguments` instance in the script.
