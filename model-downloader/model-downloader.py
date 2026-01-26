from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

model_id = "meta-llama/Llama-3.2-3B"

tokenizer_clean = AutoTokenizer.from_pretrained(
    model_id
)

model_clean = AutoModelForCausalLM.from_pretrained(
    model_id,
    cache_dir="C:/Users/farka/Projects/Diplomamunka/Models/llama3.2-3b-clean"
)

tokenizer_clean.save_pretrained("C:/Users/farka/Projects/Diplomamunka/Models/llama3.2-3b-clean")
model_clean.save_pretrained("C:/Users/farka/Projects/Diplomamunka/Models/llama3.2-3b-clean")

tokenizer_lora = AutoTokenizer.from_pretrained(
    model_id
)

model_lora = AutoModelForCausalLM.from_pretrained(
    model_id,
    cache_dir="C:/Users/farka/Projects/Diplomamunka/Models/llama3.2-3b-lora"
)

tokenizer_lora.save_pretrained("C:/Users/farka/Projects/Diplomamunka/Models/llama3.2-3b-lora")
model_lora.save_pretrained("C:/Users/farka/Projects/Diplomamunka/Models/llama3.2-3b-lora")