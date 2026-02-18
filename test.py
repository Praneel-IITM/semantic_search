import os
from openai import OpenAI

client = OpenAI(
    base_url="https://aipipe.org/openai/v1",
    api_key=os.getenv("AIPIPE_TOKEN")
)

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "Hello"}]
)

print(response.choices[0].message.content)
