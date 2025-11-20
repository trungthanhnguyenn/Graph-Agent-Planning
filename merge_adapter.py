import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import os

def merge_lora_adapter():
    base_model_name_or_path = "Qwen/Qwen2.5-3B-Instruct"
    adapter_path = "Graph-Agent-Planning/experiments/exp_1_lr1e-5_wr0.03_bs1_ga16"
    merged_model_path = "Graph-Agent-Planning/experiments/merged_model"

    print(f"Loading base model from {base_model_name_or_path}...")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name_or_path,
        torch_dtype=torch.bfloat16,
        device_map="cpu",
        trust_remote_code=True
    )

    print(f"Loading adapter from {adapter_path}...")
    model = PeftModel.from_pretrained(base_model, adapter_path)

    print("Merging adapter into the base model...")
    model = model.merge_and_unload()

    print(f"Saving merged model to {merged_model_path}...")
    os.makedirs(merged_model_path, exist_ok=True)
    model.save_pretrained(merged_model_path)

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(base_model_name_or_path, trust_remote_code=True)
    
    print(f"Saving tokenizer to {merged_model_path}...")
    tokenizer.save_pretrained(merged_model_path)

    print("Merge complete.")

if __name__ == "__main__":
    merge_lora_adapter()
