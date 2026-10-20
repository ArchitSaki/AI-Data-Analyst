
import os

from dotenv import load_dotenv
from groq import Groq

load_dotenv()


def get_groq_client() -> Groq:
    api_key = os.getenv("GROQ_API_KEY")

    if not api_key or api_key == "your_groq_api_key_here":
        raise RuntimeError(
            "Set your actual GROQ_API_KEY in the .env file."
        )

    return Groq(api_key=api_key)


def ask_groq(system_prompt: str, user_prompt: str) -> str:
    model = os.getenv("GROQ_MODEL")

    if not model or model == "your_supported_model_id":
        raise RuntimeError(
            "Set GROQ_MODEL to a model currently supported by Groq."
        )

    client = get_groq_client()

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.1,
    )

    return response.choices[0].message.content or ""
