from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments
)
from datasets import load_dataset
import torch

print(torch.cuda.is_available())   # True
print(torch.cuda.device_count())   # 1
print(torch.cuda.get_device_name(0))  # NVIDIA GeForce RTX 2060 SUPER

torch.cuda.empty_cache()
torch.cuda.reset_peak_memory_stats()

# Load dataset
dataset = load_dataset(
    "text",
    data_files={"train": r"./resources/P4-16 Language Specification.txt"}
)

# Tokenizer
tokenizer = AutoTokenizer.from_pretrained(
    r"C:/Users/farka/Projects/Diplomamunka/Models/qwen2.5-0.5b-clean"
)

tokenized_dataset = dataset.map(
    lambda x: tokenizer(x["text"], truncation=True, padding=False),
    batched=True,
    remove_columns=["text"]
)

# Chunking with overlap to reduce memory spikes
def group_texts_with_overlap(examples, block_size=512, stride=128):
    concatenated = {k: sum(examples[k], []) for k in examples.keys()}
    input_ids = concatenated["input_ids"]
    attention_mask = concatenated["attention_mask"]

    result_input_ids = []
    result_attention_mask = []

    for i in range(0, len(input_ids) - block_size + 1, block_size - stride):
        result_input_ids.append(input_ids[i : i + block_size])
        result_attention_mask.append(attention_mask[i : i + block_size])

    return {"input_ids": result_input_ids, "attention_mask": result_attention_mask}

tokenized_dataset = tokenized_dataset.map(group_texts_with_overlap, batched=True)

# Data collator
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False
)

model = AutoModelForCausalLM.from_pretrained(
    r"C:/Users/farka/Projects/Diplomamunka/Models/qwen2.5-0.5b-clean",
    device_map="auto",
    offload_folder=r"C:/Users/farka/Projects/Diplomamunka/Offload/qwen2.5-0.5b-clean",
    torch_dtype=torch.float32
)

print(model.dtype)

# Padding tokens
tokenizer.pad_token = tokenizer.eos_token
model.config.pad_token_id = tokenizer.eos_token_id

# Gradient checkpointing to save memory
model.gradient_checkpointing_enable()

# Training arguments
training_args = TrainingArguments(
    output_dir=r"C:/Users/farka/Projects/Diplomamunka/Models/qwen2.5-0.5b-pretrained-clean",
    num_train_epochs=3,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=1,
    learning_rate=2e-5,
    warmup_steps=100,
    save_steps=10_000,
    save_total_limit=2,
    logging_steps=100
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    data_collator=data_collator,
)

trainer.train()