import os
from dotenv import load_dotenv
from openai import OpenAI
from langfuse import get_client
from langfuse.openai import openai

load_dotenv()

# Initialize Langfuse and the model client
langfuse = get_client()

client = openai.OpenAI(
    api_key=os.getenv("OPENAI_API_KEY")
)

MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

POLICY = """
Company refund policy:
- Refund requests must be submitted within 30 days of purchase.
- The product must meet the company's eligibility requirements.
- If a case is not covered by this policy, ask the customer
  to contact support for confirmation.
- Do not invent exceptions or promise a refund.
"""

def ask_support_assistant(question):
    with langfuse.start_as_current_observation(
        as_type="span",
        name="customer-support-request",
        input={"question": question},
    ) as span:

        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a customer support assistant. "
                        "Answer only using the policy provided. "
                        "If the policy does not answer the question, "
                        "say that support must confirm it.\n\n"
                        + POLICY
                    ),
                },
                {"role": "user", "content": question},
            ],
        )

        answer = response.choices[0].message.content

        span.update(
            output={"answer": answer},
            metadata={
                "model": MODEL,
                "scenario": "customer-support",
            },
        )

        return answer


if __name__ == "__main__":
    questions = [
        "Can I request a refund after 10 days?",
        "Can I get a refund after 60 days?",
        "Does the policy guarantee refunds for every product?",
    ]

    try:
        for question in questions:
            print("\nQUESTION:", question)
            print("ANSWER:", ask_support_assistant(question))
    finally:
        langfuse.flush()
