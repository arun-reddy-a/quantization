import torch
from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer

path = "./llama3_1_8b_awq_int4"
tok = AutoTokenizer.from_pretrained(path, use_fast=True)
m = AutoAWQForCausalLM.from_quantized(path, device_map="auto")

prompt = "Write a poem about a cat."
inputs = tok(prompt, return_tensors="pt").to(m.device)
out = m.generate(**inputs, max_new_tokens=80)
print(tok.decode(out[0], skip_special_tokens=True))
