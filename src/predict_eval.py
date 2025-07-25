import os

import evaluate
import torch
from datasets import load_dataset
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from config import BASE_MODEL_NAME
from config import DATASET_PATH as TEST_DATA_PATH
from config import MODEL_NAME as ADAPTER_NAME


def main():
    base_model = AutoModelForCausalLM.from_pretrained(BASE_MODEL_NAME, torch_dtype=torch.bfloat16, device_map="auto")
    model = PeftModel.from_pretrained(base_model, ADAPTER_NAME)
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token
    
    eval_dataset = load_dataset("json", data_files=TEST_DATA_PATH, split="train")
    bertscore = evaluate.load("bertscore")
    
    predictions = []
    references = []

    for example in eval_dataset:
        instruction = example['instruction']
        ground_truth = example['response']
        prompt = f"### instruction:\n{instruction}\n\n### response:\n"
        inputs = tokenizer(prompt, return_tensors="pt", return_token_type_ids=False).to(model.device)
        outputs = model.generate(**inputs, max_new_tokens=128, eos_token_id=tokenizer.eos_token_id)
        generated_response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        
        predictions.append(generated_response)
        references.append(ground_truth)

    results = bertscore.compute(predictions=predictions, references=references, lang="ko")
    
    avg_f1 = sum(results['f1']) / len(results['f1'])
    
    print(f"BERTScore (F1): {avg_f1:.4f}")
    print("="*30)

if __name__ == "__main__":
    main()