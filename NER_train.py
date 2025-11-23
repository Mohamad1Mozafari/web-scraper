import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification, TrainingArguments, Trainer
from datasets import load_dataset, load_metric

# Load dataset in "conll" format
data_files = {"train": "train.txt", "validation": "valid.txt"}
dataset = load_dataset("conll2003", data_files=data_files, split=None)

# Define labels
labels = ["O", "B-PER", "I-PER", "B-LOC", "I-LOC", "B-PHONE", "I-PHONE", "B-EMAIL", "I-EMAIL"]
label2id = {l: i for i, l in enumerate(labels)}
id2label = {i: l for i, l in enumerate(labels)}

# Load tokenizer and model
model_name = "HooshvareLab/bert-base-parsbert-ner"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForTokenClassification.from_pretrained(model_name, num_labels=len(labels), id2label=id2label, label2id=label2id)

# Tokenize and align labels
def tokenize_and_align_labels(batch):
    tokenized_inputs = tokenizer(batch["tokens"], truncation=True, is_split_into_words=True)
    labels_batch = []
    for i, label in enumerate(batch["ner_tags"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(-100)
            elif word_idx != previous_word_idx:
                label_ids.append(label2id[label[word_idx]])
            else:
                # For inside tokens, use I- prefix if exists
                cur_label = label[word_idx]
                if cur_label.startswith("B-"):
                    cur_label = "I-" + cur_label[2:]
                label_ids.append(label2id[cur_label])
            previous_word_idx = word_idx
        labels_batch.append(label_ids)
    tokenized_inputs["labels"] = labels_batch
    return tokenized_inputs

tokenized_datasets = dataset.map(tokenize_and_align_labels, batched=True)

# Metrics
metric = load_metric("seqeval")

def compute_metrics(p):
    predictions, labels_ = p
    predictions = predictions.argmax(-1)
    true_labels = [[id2label[l] for l in label if l != -100] for label in labels_]
    true_predictions = [
        [id2label[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels_)
    ]
    results = metric.compute(predictions=true_predictions, references=true_labels)
    return {
        "precision": results["overall_precision"],
        "recall": results["overall_recall"],
        "f1": results["overall_f1"],
        "accuracy": results["overall_accuracy"],
    }

# Training
training_args = TrainingArguments(
    output_dir="./parsbert-ner",
    evaluation_strategy="epoch",
    learning_rate=5e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=5,
    weight_decay=0.01,
    save_total_limit=2,
    logging_dir="./logs",
    logging_steps=50,
    report_to="none",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

# Train
trainer.train()

# Save model
trainer.save_model("./parsbert-ner-finetuned")
