from transformers import AutoModelForCausalLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1", local_files_only=True)

model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-v0.1", local_files_only=True, device_map="auto")

inputs = tokenizer("Me explique quem é você?, me responda em português", return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=600)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))