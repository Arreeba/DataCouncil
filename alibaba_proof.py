
import openai
import os

# This file proves DataCouncil uses Alibaba Cloud DashScope API
# Base URL: https://dashscope-intl.aliyuncs.com/compatible-mode/v1

client = openai.OpenAI(
    api_key=os.environ.get("QWEN_API_KEY"),
    base_url="https://dashscope-intl.aliyuncs.com/compatible-mode/v1"
)

def verify_alibaba_cloud_connection():
    response = client.chat.completions.create(
        model="qwen-plus",
        messages=[
            {"role": "user",
             "content": "Reply with exactly: Alibaba Cloud DashScope connection verified."}
        ]
    )
    print(response.choices[0].message.content)
    print(f"Model: {response.model}")

if __name__ == "__main__":
    verify_alibaba_cloud_connection()
