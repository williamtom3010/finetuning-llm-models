# prompt the generation model
from huggingface_hub import login
from transformers import AutoTokenizer, AutoModelForCausalLM  
import torch
from sentence_transformers import SentenceTransformer
import os
from dotenv import load_dotenv

# Load model directly

load_dotenv()
HF_READ_TOKEN = os.getenv("HF_READ_TOKEN")
login(HF_READ_TOKEN)
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)


def load_generator_model():
    
    gen_model_path ="meta-llama/Llama-3.2-3B-Instruct"
    generator_model = AutoModelForCausalLM.from_pretrained(
        gen_model_path,
        torch_dtype=torch.float16,
        # quantization_config=bnb_config,
        device_map="auto"
    )    
    # generator_model = generator_model.to(device)
    print("Model has been loaded")

    tokenizer = AutoTokenizer.from_pretrained(gen_model_path)
    print("Tokenizer has been loaded")

    return generator_model,tokenizer

# generator_model,tokenizer = load_generator_model()