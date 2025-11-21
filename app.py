"""
app.py - Flask web UI for LLM Q&A
Run:
  export FLASK_APP=app.py
  flask run
or
  python app.py

Environment:
  - For OpenAI: set OPENAI_API_KEY

Notes:
  - Very small, single-file Flask app that renders templates/index.html
"""

import os
from flask import Flask, render_template, request, redirect, url_for, flash
import re
import textwrap

app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET", "dev-secret-key")  # change in production

# Preprocess (same rules as CLI)
def preprocess_question(question: str) -> (str, list):
    q = question.strip()
    q_lower = q.lower()
    q_clean = re.sub(r'[^\w\s]', ' ', q_lower)
    q_clean = re.sub(r'\s+', ' ', q_clean).strip()
    tokens = q_clean.split()
    return q_clean, tokens

def build_prompt(processed_question: str) -> str:
    return textwrap.dedent(f"""
    You are a helpful assistant. Answer concisely.

    Question:
    {processed_question}

    Answer:
    """).strip()

def call_llm(prompt: str, max_tokens: int = 256, temperature: float = 0.2):
    # Use OpenAI by default. Swap this for other providers if needed.
    try:
        import openai
    except Exception as e:
        raise RuntimeError("openai package required. Install with `pip install openai`.") from e
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("Please set OPENAI_API_KEY environment variable.")
    openai.api_key = api_key

    # Use ChatCompletion first, fallback if needed
    try:
        resp = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=temperature,
        )
        text = resp["choices"][0]["message"]["content"].strip()
        return {"text": text, "raw": resp}
    except Exception:
        resp = openai.Completion.create(
            model="text-davinci-003",
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        text = resp["choices"][0]["text"].strip()
        return {"text": text, "raw": resp}

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        question = request.form.get("question", "").strip()
        if not question:
            flash("Please enter a question.", "warning")
            return redirect(url_for("index"))

        processed, tokens = preprocess_question(question)
        prompt = build_prompt(processed)
        try:
            llm_result = call_llm(prompt)
            answer = llm_result["text"]
            raw = llm_result["raw"]
        except Exception as e:
            answer = ""
            raw = {"error": str(e)}
            flash(f"Error calling LLM: {e}", "danger")

        return render_template("index.html",
                               question=question,
                               processed=processed,
                               tokens=tokens,
                               prompt=prompt,
                               answer=answer,
                               raw=raw)
    return render_template("index.html")

if __name__ == "__main__":
    # Use debug=False for production
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", 8501)), debug=True)
