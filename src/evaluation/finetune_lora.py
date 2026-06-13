import json
import os
import sys
import torch
sys.path.insert(0, '/root/project')

from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model, TaskType
from torch.utils.data import Dataset

MODEL_PATH   = "/root/models/qwen2.5-7b-hf/Qwen/Qwen2___5-7B-Instruct"
OUTPUT_DIR   = "/root/project/models/qwen2.5-q4-finetuned"
DATASET_PATH = "/root/project/data/finetune/alpaca_dataset.json"

print("="*50)
print("  NEXORA LoRA Fine-tuning")
print("="*50)

with open(DATASET_PATH, 'r', encoding='utf-8') as f:
    raw_data = json.load(f)
print(f"\n Dataset loaded — {len(raw_data)} examples")

def format_prompt(example):
    return f"""### Instruction:
{example['instruction']}

### Input:
{example['input']}

### Response:
{example['output']}"""

class AlpacaDataset(Dataset):
    def __init__(self, data, tokenizer, max_length=512):
        self.data       = data
        self.tokenizer  = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text     = format_prompt(self.data[idx])
        encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt"
        )
        input_ids = encoding["input_ids"].squeeze()
        return {"input_ids": input_ids, "labels": input_ids.clone()}

print("\n............... Loading tokenizer............")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
print(" Tokenizer loaded")

print("\n............ Loading model..............")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.float16,
    device_map="auto",
    load_in_8bit=True,
    trust_remote_code=True
)
print(" Model loaded")

lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=8,
    lora_alpha=32,
    lora_dropout=0.1,
    target_modules=["q_proj", "v_proj"]
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

dataset = AlpacaDataset(raw_data, tokenizer)
print(f"\n Dataset ready — {len(dataset)} examples")

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=3,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    learning_rate=2e-4,
    fp16=True,
    logging_steps=5,
    save_steps=50,
    save_total_limit=2,
    warmup_steps=10,
    report_to="none"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
)

print("\n...... Starting fine-tuning......")
trainer.train()

os.makedirs(OUTPUT_DIR, exist_ok=True)
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print(f"\n Model saved to {OUTPUT_DIR}")
print("="*50)
