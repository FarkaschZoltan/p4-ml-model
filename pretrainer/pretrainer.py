from transformers import AutoTokenizer, AutoModelForMaskedLM, DataCollatorForLanguageModeling, Trainer, TrainingArguments
from datasets import load_dataset

# Load domain-specific text (replace with your own corpus)
dataset = load_dataset("text", data_files={"train": "resources/P4-16 Language Specification.txt"})

# Tokenizer
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
tokenized_dataset = dataset.map(lambda x: tokenizer(x["text"], truncation=True, padding=True), batched=True)

# Data collator for Masked Language Modeling
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=True, mlm_probability=0.15)

# Load pre-trained model
model = AutoModelForMaskedLM.from_pretrained("bert-base-uncased")

# Training arguments
training_args = TrainingArguments(
    output_dir="./domain_pretrained_model",
    overwrite_output_dir=True,
    num_train_epochs=3,
    per_device_train_batch_size=8,
    save_steps=10_000,
    save_total_limit=2,
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    data_collator=data_collator,
)

# Pretrain the model
trainer.train()