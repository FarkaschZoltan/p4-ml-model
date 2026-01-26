from transformers import AutoTokenizer, AutoModelForCausalLM

model_id = "meta-llama/Llama-3.2-3B"

tokenizer_clean = AutoTokenizer.from_pretrained(
    model_id,
    use_auth_token=True
)

model_clean = AutoModelForCausalLM.from_pretrained(
    model_id,
    use_auth_token=True,
    cache_dir="C:/Users/farka/OneDrive/Egyetem/MSC/diplomamunka/Models/llama3.2-3b-clean"
)

tokenizer_clean.save_pretrained("C:/Users/farka/OneDrive/Egyetem/MSC/diplomamunka/Models/llama3.2-3b-clean")
model_clean.save_pretrained("C:/Users/farka/OneDrive/Egyetem/MSC/diplomamunka/Models/llama3.2-3b-clean")

tokenizer_lora = AutoTokenizer.from_pretrained(
    model_id,
    use_auth_token=True
)

model_lora = AutoModelForCausalLM.from_pretrained(
    model_id,
    use_auth_token=True,
    cache_dir="C:/Users/farka/OneDrive/Egyetem/MSC/diplomamunka/Models/llama3.2-3b-lora"
)

tokenizer_lora.save_pretrained("C:/Users/farka/OneDrive/Egyetem/MSC/diplomamunka/Models/llama3.2-3b-lora")
model_lora.save_pretrained("C:/Users/farka/OneDrive/Egyetem/MSC/diplomamunka/Models/llama3.2-3b-lora")