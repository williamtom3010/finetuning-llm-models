import torch
from transformers import pipeline
import models


## Generate model response
# good for llama 3.1-8b base model(cpt version)

def generate_response(user_question, model, tokenizer, max_new_tokens=100):

    prompt = (
        f"<s>[INST] Question:\n{user_question}\n\n [/INST]"
        "### Answer:\n" 
    )


    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.6,
            top_k=30,
            top_p=0.9,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
            do_sample=True
        )

    # Decode and extract only the generated answer part
    full_output = tokenizer.decode(output[0], skip_special_tokens=True)
    generated_answer = full_output.split("### Answer:\n")[-1].strip()
    # generated_answer = full_output.split("Answer:\n")[-1].strip()

    return generated_answer



# good for llama 3.1-8b instruct model

def generate_response(user_question, model, tokenizer):
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        device_map="auto",
        torch_dtype=torch.float16
    )

    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user",   "content": user_question}
    ]
                                                                       
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    out = pipe(prompt, max_new_tokens=300, do_sample=False, temperature=0.7, top_k=30, top_p=0.9)
    text = out[0]["generated_text"]
    response = text[len(prompt):].strip()
    # print(prompt)
    return response


model, tokenizer = models.load_generator_model()
user_question = "What should happen at the first claimant meeting?"
response = generate_response(user_question, model, tokenizer)
print("\nGenerated Answer:\n", response)