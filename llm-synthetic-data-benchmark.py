import json
import time
import torch
from openai import OpenAI
from google.colab import userdata
from huggingface_hub import login
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, pipeline

LLAMA = "meta-llama/Llama-3.1-8B-Instruct"
OPENAI_MODEL = "gpt-4.1-mini"

hf_token = userdata.get('HF_TOKEN')
login(hf_token, add_to_git_credential=True)

openai_api_key = userdata.get('OPENAI_API_KEY')
openai = OpenAI(api_key=openai_api_key)


system_message = """
You are a synthetic data generation engine that creates realistic, structured employee records for a large U.S. bank.
Output ONLY valid JSON.
"""

user_prompt = """
Generate 3 synthetic employee records for a U.S. bank.
Return ONLY the JSON array.
"""

messages = [
    {"role": "system", "content": system_message},
    {"role": "user", "content": user_prompt}
]


# =========================
# QUANTIZATION (KEY IDEA)
# =========================
# Shrinks model weights from 16/32-bit → 4-bit
# → drastically reduces GPU memory usage
# → slight drop in output quality

quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16
)


# =========================
# TOKENIZER (VERY IMPORTANT)
# =========================
# Converts text → token IDs (numbers)
# Neural networks do NOT understand text, only numbers
#
# Example:
# "Bank analyst" → [1284, 9382, 552]
#
# These token IDs are what the model actually processes

tokenizer = AutoTokenizer.from_pretrained(LLAMA)
tokenizer.pad_token = tokenizer.eos_token


# =========================
# MODEL (NEURAL NETWORK)
# =========================
# Loads the trained weights of the transformer model
#
# CausalLM = predicts NEXT token given previous tokens
#
# Core generation loop:
# 1. Read tokens
# 2. Predict next token probabilities
# 3. Pick one token
# 4. Append it
# 5. Repeat

model = AutoModelForCausalLM.from_pretrained(
    LLAMA,
    quantization_config=quant_config,
    device_map="auto"
)

memory = model.get_memory_footprint() / 1e6
print(f"Memory footprint: {memory:,.1f} MB")

print(f"Printing Model")
print(f"{model}")


pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer
)


prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
{system_message}<|eot_id|><|start_header_id|>user<|end_header_id|>
{user_prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>
"""


# =========================
# GENERATION (HF MODEL)
# =========================
# Internally:
# - prompt → tokens (tensor)
# - passed through transformer layers
# - outputs probability distribution over vocab
# - sampling picks next token

start_hf = time.time()

hf_result = pipe(
    prompt,
    max_new_tokens=800,
    temperature=0.8,
    do_sample=True
)

hf_time = time.time() - start_hf

hf_text = hf_result[0]["generated_text"][len(prompt):].strip()

print("HF RESULT:")
print(hf_text)
print(f"HF time: {hf_time:.2f}s")


# =========================
# OPENAI MODEL
# =========================
# Same concept, but model is hosted
# You only send messages, OpenAI handles inference

start_openai = time.time()

openai_response = openai.chat.completions.create(
    model=OPENAI_MODEL,
    messages=messages
)

openai_time = time.time() - start_openai

openai_text = openai_response.choices[0].message.content

print("\nOPENAI RESULT:")
print(openai_text)
print(f"OpenAI time: {openai_time:.2f}s")


# =========================
# VALIDATION (IMPORTANT FOR PROJECT)
# =========================

required_keys = {
    "employee_id",
    "first_name",
    "last_name",
    "email",
    "job_title",
    "department",
    "location",
    "years_of_experience",
    "salary_usd",
    "employment_type",
    "manager_id",
    "hire_date",
    "skills",
    "performance_rating"
}

def validate(text):
    try:
        data = json.loads(text)
    except:
        return {"valid_json": False}

    if not isinstance(data, list):
        return {"valid_json": True, "schema_ok": False}

    for r in data:
        if set(r.keys()) != required_keys:
            return {"valid_json": True, "schema_ok": False}

    return {"valid_json": True, "schema_ok": True, "records": len(data)}


print("\nCOMPARISON:")
print("HF:", validate(hf_text))
print("OpenAI:", validate(openai_text))llm-synthetic-data-benchmark
