import os
import json
import torch
import pandas as pd
from dotenv import load_dotenv
from trl import SFTTrainer
from huggingface_hub import login
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, TaskType
from transformers import BitsAndBytesConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments



# Load HF TOKENS
load_dotenv()
HF_READ_TOKEN = os.getenv("HF_READ_TOKEN")
login(HF_READ_TOKEN)


csv_path = "data_files/new_augemneted_final_dataset.csv"
json_path = "json_folder/auditor.jsonl"

# from datasets import load_dataset
# ds = load_dataset("harleygilpin/soc-audit-11k")
# print(ds)
# print(ds['train'][0])

# Load your CSV file
df = pd.read_csv(csv_path)


# add chat template to the dataset
system_prompt = "You are a helpful assistant."

with open(json_path, 'w', encoding='utf-8') as f:
    for _, row in df.iterrows():
        # user_msg = row['Question']
        # assistant_resp = row['Answer']

        system_prompt = row['system']
        user_msg = row['instruction']
        assistant_resp = row['response']
        
        text = (
            "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
            f"{system_prompt}<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n"
            f"{user_msg}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
            f"{assistant_resp}<|eot_id|>"
        )
        json.dump({"text": text}, f)
        f.write("\n")


# Load the dataset
dataset = load_dataset('json', data_files=json_path)

print(dataset)

print(dataset['train']['text'][0])

model_path = "williamtom-3010/llama-it_claim_adjuster-sft"



# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_path)
tokenizer.pad_token = tokenizer.eos_token  # Ensure proper padding

# Tokenize the dataset
def tokenize_function(example):
    return tokenizer(example["text"], 
                     truncation=True, 
                     padding='max_length', 
                     max_length=512)
    

tokenized_dataset = dataset.map(tokenize_function, batched=True)

# Configure 4-bit quantization
# bnb_config = BitsAndBytesConfig(
#     load_in_4bit=True,
#     bnb_4bit_use_double_quant=True,
#     bnb_4bit_quant_type="nf4",
#     bnb_4bit_compute_dtype=torch.bfloat16
# )


# Load the base model
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.float16,
    # quantization_config=bnb_config,
    device_map="auto"
)

print(model)

# Define LoRA configuration
lora_config = LoraConfig(
    r=64,
    lora_alpha=128,
    # target_modules=["q_proj", "v_proj"],
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    # task_type=TaskType.CAUSAL_LM
)

# Apply LoRA to the model
model = get_peft_model(model, lora_config)

print(tokenized_dataset)
print(tokenized_dataset['train'])


output_dir = "fine_tune_models/llama-it_auditor-sft"


training_args = TrainingArguments(
    output_dir=output_dir,
    per_device_train_batch_size=11,
    gradient_accumulation_steps=5,
    num_train_epochs=3,
    learning_rate=2e-4,
    logging_steps=10,
    save_steps=100,
    save_total_limit=2,
    report_to="none",
    save_strategy="steps",    
    bf16=False,
    fp16=True,
    warmup_steps=50,
    lr_scheduler_type="cosine",
    # logging_dir="./logs",
    optim="adamw_torch"
)

# Initialize the trainer
trainer = SFTTrainer(
    model=model,
    train_dataset=tokenized_dataset['train'],
    args=training_args,
    # peft_config=lora_config,
    # max_seq_length=512
)

# Start training
trainer.train()

# --- Save LoRA Adapter ---
model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)
print(f"✅ LoRA fine-tuned model saved to {output_dir}")

