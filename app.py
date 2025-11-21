# app.py
import os
import re
from flask import Flask, request, render_template, jsonify
from openai import OpenAI

app = Flask(__name__)

def preprocess_question(question: str) -> str:
    q = question.lower()
    q = re.sub(r"[^\w\s]", "", q)
    q = re.sub(r"\s+", " ", q).strip()
    return q

def construct_prompt(processed_q: str) -> str:
    return f"You are a helpful assistant. Answer concisely.\nQuestion: {processed_q}\nAnswer:"

def call_llm(prompt: str) -> dict:
    client = OpenAI()  # requires OPENAI_API_KEY in environment
    resp = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.2,
        max_tokens=300,
    )
    content = resp.choices[0].message.content.strip()
    return {"answer": content, "raw": resp.model_dump()}

@app.route("/", methods=["GET", "POST"])
def index():
    answer = None
    raw = None
    processed = None
    prompt = None
    error = None

    if request.method == "POST":
        user_q = request.form.get("question", "").strip()
        processed = preprocess_question(user_q)
        prompt = construct_prompt(processed)
        try:
            result = call_llm(prompt)
            answer = result["answer"]
            raw = result["raw"]
        except Exception as e:
            error = f"Error calling LLM: {e}"

    return render_template(
        "index.html",
        answer=answer,
        raw=raw,
        processed=processed,
        prompt=prompt,
        error=error
    )

@app.route("/api/ask", methods=["POST"])
def api_ask():
    data = request.get_json(force=True)
    user_q = data.get("question", "")
    processed = preprocess_question(user_q)
    prompt = construct_prompt(processed)
    try:
        result = call_llm(prompt)
        return jsonify({
            "processed": processed,
            "prompt": prompt,
            "answer": result["answer"],
            "raw": result["raw"],
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
