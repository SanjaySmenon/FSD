import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel, PeftConfig
from transformers import BitsAndBytesConfig

base_model_name = "Qwen/Qwen2.5-Coder-3B-Instruct"
adapter_path = "/content/drive/MyDrive/full stack model"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True
)

# Load base model
model = AutoModelForCausalLM.from_pretrained(
    base_model_name,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True
)

# Attach LoRA adapter
model = PeftModel.from_pretrained(model, adapter_path)

tokenizer = AutoTokenizer.from_pretrained(adapter_path)

model.eval()

print("✅ Model reloaded successfully")
