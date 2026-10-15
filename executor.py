import os
from dotenv import load_dotenv
from openai import OpenAI

# Load environment variables from .env file
load_dotenv()

# Configuration
MODEL_NAME = "gpt-4o-mini"
api_key = os.getenv("OPENAI_API_KEY")

if not api_key: 
    raise RuntimeError("OPENAI_API_KEY not found. Check your .env file.")

# Initialize OpenAI Client
client = OpenAI(api_key=api_key)

def generate_response(question, context):
    system_prompt = """You are an enterprise question-answering assistant.Answer using only the information provided in the context.Do not invent facts or add unsupported details.If the context does not contain enough information,say that the information is not available.Be concise and answer the user's question directly."""
    
    user_prompt = f"""Context:{context}Question:{question}Answer:"""
    
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {
                "role": "system",
                "content": system_prompt,
            },
            {
                "role": "user",
                "content": user_prompt,
            },
        ],
        temperature=0,
    )
    
    return response.choices[0].message.content.strip()