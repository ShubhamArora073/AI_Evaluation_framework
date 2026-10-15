import os

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def execute_case(case):
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are an enterprise policy assistant. "
                    "Answer using only the supplied context. "
                    "Do not invent policies or facts. "
                    "If the context is insufficient, clearly say so."
                )
            },
            {
                "role": "user",
                "content": (
                    f"Context:\n{case['context']}\n\n"
                    f"Question:\n{case['question']}"
                )
            }
        ]
    )

    return response.choices[0].message.content or ""
