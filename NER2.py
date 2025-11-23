import torch
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    TrainingArguments,
    Trainer,
)
from datasets import load_dataset
import evaluate

# -----------------------------
# 1. Load model and tokenizer
# -----------------------------
model_name = "HooshvareLab/bert-base-parsbert-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForTokenClassification.from_pretrained(model_name, num_labels=9)  # adjust later if needed

# -----------------------------
# 2. Load a sample dataset
# -----------------------------
dataset = load_dataset("wikiann", "fa")

label_list = dataset["train"].features["ner_tags"].feature.names
num_labels = len(label_list)
id2label = {i: label for i, label in enumerate(label_list)}
label2id = {label: i for i, label in enumerate(label_list)}

model.config.id2label = id2label
model.config.label2id = label2id

# -----------------------------
# 3. Tokenization function (fixed)
# -----------------------------
def tokenize_and_align_labels(batch):
    tokenized_inputs = tokenizer(
        batch["tokens"],
        truncation=True,
        is_split_into_words=True,
        padding="max_length",   # ✅ ensures consistent length
        max_length=128
    )

    labels = []
    for i, label in enumerate(batch["ner_tags"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(-100)
            elif word_idx != previous_word_idx:
                label_ids.append(label[word_idx])
            else:
                label_ids.append(label[word_idx] if label[word_idx] == 0 else label[word_idx])  # same label for subwords
            previous_word_idx = word_idx
        labels.append(label_ids)

    tokenized_inputs["labels"] = labels
    return tokenized_inputs

# -----------------------------
# 4. Preprocess datasets
# -----------------------------
tokenized_datasets = dataset.map(tokenize_and_align_labels, batched=True, remove_columns=dataset["train"].column_names)

# -----------------------------
# 5. Load metric
# -----------------------------
metric = evaluate.load("seqeval")

def compute_metrics(p):
    predictions, labels = p
    predictions = torch.argmax(torch.tensor(predictions), dim=2)

    true_predictions = [
        [id2label[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [id2label[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    results = metric.compute(predictions=true_predictions, references=true_labels)
    return {
        "precision": results["overall_precision"],
        "recall": results["overall_recall"],
        "f1": results["overall_f1"],
        "accuracy": results["overall_accuracy"],
    }

# -----------------------------
# 6. Training setup
# -----------------------------
training_args = TrainingArguments(
    output_dir="./parsbert-ner",
    evaluation_strategy="epoch",
    learning_rate=5e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=2,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=50,
    report_to="none"  # ✅ disables wandb logging
)

# -----------------------------
# 7. Trainer
# -----------------------------
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"].select(range(1000)),  # subset for faster testing
    eval_dataset=tokenized_datasets["validation"].select(range(200)),
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

# -----------------------------
# 8. Train
# -----------------------------
trainer.train()
