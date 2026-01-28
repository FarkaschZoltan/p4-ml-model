from transformers import AutoTokenizer, AutoModelForCausalLM

model_id = "Qwen/Qwen2.5-0.5B"

tokenizer_clean = AutoTokenizer.from_pretrained(
    model_id
)

model_clean = AutoModelForCausalLM.from_pretrained(
    model_id,
    cache_dir="C:/Users/farka/Projects/Diplomamunka/Models/qwen2.5-0.5b-clean"
)

tokenizer_clean.save_pretrained("C:/Users/farka/Projects/Diplomamunka/Models/qwen2.5-0.5b-clean")
model_clean.save_pretrained("C:/Users/farka/Projects/Diplomamunka/Models/qwen2.5-0.5b-clean")

tokenizer_lora = AutoTokenizer.from_pretrained(
    model_id
)

model_lora = AutoModelForCausalLM.from_pretrained(
    model_id,
    cache_dir="C:/Users/farka/Projects/Diplomamunka/Models/qwen2.5-0.5b-lora"
)

tokenizer_lora.save_pretrained("C:/Users/farka/Projects/Diplomamunka/Models/qwen2.5-0.5b-lora")
model_lora.save_pretrained("C:/Users/farka/Projects/Diplomamunka/Models/qwen2.5-0.5b-lora")
