#%%
### test file 
"""
example of LLAMA 3 chatbot with gradio available 
https://github.com/johnsonice/HuggingFace_Demos/blob/main/examples/Gradio_Inference/Example_LLAMA3_8B.py
"""


import transformers
import torch

#%%
model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
cache_dir = "/root/data/hf_cache"
hf_access_token = input("Please enter the hf key: ") ## hf access token 

pipeline = transformers.pipeline(
    "text-generation",
    model=model_id,
    model_kwargs={"torch_dtype": torch.bfloat16}, ## load with bf16
    device_map="auto",
    #cache_dir = cache_dir,
    token = hf_access_token
)
messages = [
    {"role": "system", "content": "You are a pirate chatbot who always responds in pirate speak!"},
    {"role": "user", "content": "Who are you?"},
]

terminators = [
    pipeline.tokenizer.eos_token_id,
    pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>")
]
outputs = pipeline(
    messages,
    max_new_tokens=256,
    eos_token_id=terminators,
    do_sample=True,
    temperature=0.6,
    top_p=0.9,
)

print(outputs[0]["generated_text"][-1])

