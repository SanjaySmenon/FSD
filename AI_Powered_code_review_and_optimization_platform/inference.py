# ----------------------------------------------------------
# 🔟 Inference Function (Qwen Chat Style)
# ----------------------------------------------------------

def generate_review(code, request="Review and optimize the code"):

    messages = [
        {
            "role": "system",
            "content": "You are a senior software engineer. Always explain the issue clearly and then provide an improved version of the code."
        },
        {
            "role": "user",
            "content": f"{request}\n\nCode:\n{code}"
        }
    ]

    # Apply Qwen chat template
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=300,
            temperature=0.4,              # Lower = more structured
            top_p=0.9,
            repetition_penalty=1.1,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )

    # Remove prompt tokens
    generated_tokens = outputs[0][inputs["input_ids"].shape[-1]:]

    response = tokenizer.decode(
        generated_tokens,
        skip_special_tokens=True
    ).strip()

    return response


# ----------------------------------------------------------
# 🔥 Test Example
# ----------------------------------------------------------

print("\n===== SAMPLE OUTPUT =====\n")
print(generate_review("def divide(a,b): return a/b"))
