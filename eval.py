import torch
from config import Config

def evaluate_model(model, tokenizer):
    prompts = [
        "Write a short poem about the sea.",
        "Give me a recipe for chocolate cake.",
        "Explain the theory of relativity in simple terms."
    ]

    print("\n --- Running Evaluation:")
    for prompt in prompts:
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        input_token_len = inputs.input_ids.shape[1]
        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=Config.MAX_OUTPUT_TOKEN,
                do_sample=True,
                top_p=Config.TOP_P,
                temperature=Config.TEMPERATURE
            )
        decoded = tokenizer.decode(output[0][input_token_len:], skip_special_tokens=True)
        decoded = decoded.strip()
        print(f"\nPrompt: {prompt}\nResponse: {decoded}")