# ----------------------------------------------------------
# 🔟 Inference (Hardcoded Example)
# ----------------------------------------------------------

def generate_review(code):

    messages = [
        {
            "role": "system",
            "content": "You are a senior software engineer. Explain the issue clearly and then provide an improved version of the code."
        },
        {
            "role": "user",
            "content": f"Review and improve the following code:\n\n{code}"
        }
    ]

    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=400,
            temperature=0.4,
            top_p=0.9,
            repetition_penalty=1.1,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )

    generated_tokens = outputs[0][inputs["input_ids"].shape[-1]:]

    response = tokenizer.decode(
        generated_tokens,
        skip_special_tokens=True
    ).strip()

    return response


# ----------------------------------------------------------
# 🔥 Hardcoded Test Example
# ----------------------------------------------------------

example_code = """
def sum_list(arr):
    total = 0
    for i in range(len(arr)):
        total += arr[i]
    return total
"""

print("\n===== SAMPLE OUTPUT =====\n")
print(generate_review(example_code))
