# base model path
model_id = "williamtom-3010/llama-it_claim_adjuster-sft"



## Loading lora+base model seperately

# Define the lora adaptor path

output_dir = "fine_tune_models/llama-it_auditor-sft"
model_path = output_dir

from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import torch
import os


from dotenv import load_dotenv
from huggingface_hub import login

# Load HF TOKENS
load_dotenv()
HF_READ_TOKEN = os.getenv("HF_READ_TOKEN")
HF_WRITE_TOKEN  = os.getenv("HF_WRITE_TOKEN")
# read access token
login(HF_READ_TOKEN)


base_model = AutoModelForCausalLM.from_pretrained(
                                                  model_id,
                                                  torch_dtype=torch.bfloat16,
                                                  device_map="auto"
                                                  )


# Load the fine-tuned model with LoRA adapters
model = PeftModel.from_pretrained(base_model, model_path)

# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.pad_token = tokenizer.eos_token  # Ensure correct padding


# Move model to GPU if available
# device = "cuda" if torch.cuda.is_available() else "cpu"
# model.to(device)
model.eval()
print("Model loaded successfully!")

## Merging lora adaptors with base model
# First model in hugging face

merged_model_path = "merged_models/llama-it_auditor-sft"
model_1 = model.merge_and_unload()
# Save the merged model
model_1.save_pretrained(merged_model_path)
# Save tokenizer (optional, but good practice)
tokenizer_1 = AutoTokenizer.from_pretrained(model_id)
tokenizer_1.save_pretrained(merged_model_path)

print("✅ Merged model saved successfully!")


## Load model

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# To upload model to hugging face use this configuration

# new_model = "williamtom-3010/llama-it_claim_adjuster-sft"
merged_model_path = "merged_models/llama-it_auditor-sft"

model = AutoModelForCausalLM.from_pretrained(
                                             merged_model_path, 
                                             torch_dtype=torch.bfloat16, 
                                             device_map="auto"
                                             )
# model = model.to(device)

tokenizer = AutoTokenizer.from_pretrained(merged_model_path)

# model = AutoModelForCausalLM.from_pretrained(
#                                               new_model,
#                                               torch_dtype=torch.float16,
#                                               device_map="auto"
#                                               )
# # model = model.to(device)
# tokenizer = AutoTokenizer.from_pretrained(new_model)

# print("Model has been Loaded!")


## Push model to hugging face
from huggingface_hub import login
# write access token
login(HF_WRITE_TOKEN)

# Push to hub
model.push_to_hub("williamtom-3010/llama-3.1_it_auditor-sft")
tokenizer.push_to_hub("williamtom-3010/llama-3.1_it_auditor-sft")
