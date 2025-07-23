import torch
from datasets import load_dataset
from huggingface_hub import login
from peft import LoraConfig
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          BitsAndBytesConfig)
from trl import SFTConfig, SFTTrainer

from config import BASE_MODEL_NAME, DATASET_PATH, HF_TOKEN, MODEL_NAME


def main():
    login(token=HF_TOKEN)

    # load dataset
    raw_dataset = load_dataset("json", data_files=DATASET_PATH, split="train")

    # quantization config
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16
    )

    # peft config
    peft_config = LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM"
    )

    # load model
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_NAME, quantization_config=quantization_config, device_map="auto"
    )

    # load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    # training args
    training_args = SFTConfig(
        output_dir=MODEL_NAME,
        max_length=512,
        num_train_epochs=3,
        per_device_train_batch_size=2,
        learning_rate=2e-4,
        logging_steps=1,
        push_to_hub=True,
        fp16=True
    )

    # trainer
    trainer = SFTTrainer(
        model=model,
        train_dataset=raw_dataset,
        dataset_text_field="text",
        args=training_args,
        peft_config=peft_config,
        tokenizer=tokenizer,
    )

    trainer.train()

    trainer.push_to_hub()

if __name__ == "__main__":
    main()