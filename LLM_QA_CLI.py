#!/usr/bin/env python3

import os
import argparse
import re
import json
import textwrap
from typing import Tuple

# --- basic preprocessing ---
def preprocess_question(question: str) -> Tuple[str, list]:
    """Lowercase, remove punctuation, and tokenize (simple split)."""
    q = question.strip()
    q_lower = q.lower()
    # remove punctuation except question mark if you want
    q_clean = re.sub(r'[^\w\s]', ' ', q_lower)
    # collapse whitespace
    q_clean = re.sub(r'\s+', ' ', q_clean).strip()
    tokens = q_clean.split()
    return q_clean, tokens

# --- prompt crafting ---
def build_prompt(processed_question: str) -> str:
    """Construct a clear prompt for the LLM to answer the question."""
    prompt = textwrap.dedent(f"""
    You are a helpful assistant for answering a single natural-language question.
    Provide a concise, accurate answer. If the question asks for steps, enumerate them.
    If you are unsure, explain why and list assumptions.

    Question:
    {processed_question}

    Answer:
    """).strip()
    return prompt

# --- LLM call (OpenAI example) ---
def call_llm(prompt: str, max_tokens: int = 256, temperature: float = 0.2) -> dict:
    """
    Call the chosen LLM API and return the parsed result.
    This uses OpenAI's `openai` Python client by default (text-davinci-style or gpt-3.5/4 chat).
    """
    # Lazy import so the file can still be read even if openai isn't installed
    try:
        import openai
    except Exception as e:
        raise RuntimeError("openai package required for the default provider. Install with `pip install openai`.") from e

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("Please set the OPENAI_API_KEY environment variable.")

    openai.api_key = api_key

    # Use ChatCompletion if available and you prefer a chat-style model:
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4o-mini" if "gpt-4o-mini" in openai.Model.list()["data"][0].get("id", "") else "gpt-3.5-turbo",
            messages=[{"role":"user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=temperature,
        )
        # standardize output
        text = response["choices"][0]["message"]["content"].strip()
        return {"text": text, "raw": response}
    except Exception:
        # Fallback to Completion API (older)
        response = openai.Completion.create(
            model="text-davinci-003",
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        text = response["choices"][0]["text"].strip()
        return {"text": text, "raw": response}

# --- CLI logic ---
def main():
    parser = argparse.ArgumentParser(description="LLM Q&A CLI")
    parser.add_argument("--q", type=str, help="Question to ask the LLM (optional). If omitted, interactive prompt will be used.")
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--temp", type=float, default=0.2, help="LLM temperature (creativity).")
    args = parser.parse_args()

    if args.q:
        question = args.q
    else:
        print("Enter your question (single line). Press Enter to send.")
        question = input("> ").strip()
        if not question:
            print("No question provided. Exiting.")
            return

    processed_question, tokens = preprocess_question(question)
    prompt = build_prompt(processed_question)

    print("\n--- Processed Question ---")
    print(processed_question)
    print("\n--- Tokens (first 40) ---")
    print(tokens[:40])

    print("\nSending to LLM...")
    try:
        result = call_llm(prompt, max_tokens=args.max_tokens, temperature=args.temp)
    except Exception as e:
        print("Error calling LLM API:", e)
        return

    print("\n--- LLM Answer ---")
    print(result["text"])
    # Optionally show raw JSON (commented out)
    # print("\n--- Raw Response JSON ---")
    # print(json.dumps(result["raw"], indent=2, default=str))

if __name__ == "__main__":
    main()
