import os
from dotenv import load_dotenv
from huggingface_hub import InferenceClient

load_dotenv()

token = os.getenv("HF_TOKEN")

print("Token loaded:", bool(token))

client = InferenceClient(
    provider="auto",
    api_key=token
)

model = "deepseek-ai/DeepSeek-V3-0324"

print("Testing model:", model)
print("Provider: auto")
print()

try:
    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": "Say hello in one short sentence."
            }
        ],
        max_tokens=50,
        temperature=0.3
    )

    print("SUCCESS!")
    print()
    print(response.choices[0].message.content)

except Exception as e:
    print()
    print("ERROR:")
    print(repr(e))