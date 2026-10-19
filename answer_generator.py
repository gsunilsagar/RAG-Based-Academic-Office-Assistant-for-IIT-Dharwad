
import os
import requests
from dotenv import load_dotenv

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

API_URL = (
    "https://generativelanguage.googleapis.com/v1beta/models/"
    "gemini-3.5-flash-lite:generateContent"
)


def local_fallback(passages):
    if not passages:
        return "I could not find relevant information in the supplied documents."

    best = passages[0]

    return (
        "AI generation is unavailable. Most relevant information "
        "from your documents:\n\n"
        + best["text"]
        + "\n\nSource: "
        + best["source"]
        + ", page "
        + str(best["page"])
    )


def generate_answer(question, passages):
    if not passages:
        return "I could not find relevant information in the supplied documents."

    if not GEMINI_API_KEY:
        print("GEMINI_API_KEY is missing from .env.")
        return local_fallback(passages)

    context = "\n\n".join(
        "Source: {0}, page {1}\n{2}".format(
            item["source"], item["page"], item["text"]
        )
        for item in passages
    )

    payload = {
        "system_instruction": {
            "parts": [{
                "text": (
                    "You are an academic office assistant for IIT Dharwad. "
                    "Answer only from the supplied context. "
                    "Never invent dates, rules, or procedures. "
                    "Cite the source filename and page number. "
                    "If the context is insufficient, say so."
                )
            }]
        },
        "contents": [{
            "parts": [{
                "text": (
                    "Question: " + question
                    + "\n\nSupplied context:\n" + context
                )
            }]
        }],
        "generationConfig": {
            "temperature": 0.1,
            "maxOutputTokens": 350
        }
    }

    try:
        response = requests.post(
            API_URL,
            params={"key": GEMINI_API_KEY},
            headers={"Content-Type": "application/json"},
            json=payload,
            timeout=60
        )

        if response.status_code != 200:
            print(
                "Gemini API error (HTTP "
                + str(response.status_code)
                + "): " + response.text[:300]
            )
            return local_fallback(passages)

        result = response.json()
        return result["candidates"][0]["content"]["parts"][0]["text"].strip()

    except (requests.RequestException, KeyError, IndexError, ValueError) as error:
        print("Gemini request failed:", str(error))
        return local_fallback(passages)
