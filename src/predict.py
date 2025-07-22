import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from config import MODEL_NAME


def main():
    print(f"Loading fine-tuned model ({MODEL_NAME})...")
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    
    prompt = "### Instruction:\nHow are you?\n\n### Response:\n"
    
    print("="*30)
    print(f"Input prompt:\n{prompt}")
    print("="*30)
    
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(**inputs, max_new_tokens=50)
    result = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    print(f"Model generation result:\n{result}")

if __name__ == "__main__":
    main()