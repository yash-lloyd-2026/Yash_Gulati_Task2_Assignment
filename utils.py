import re
import os
import numpy as np
import pandas as pd
from groq import Groq
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from config import GROQ_API_KEY, GROQ_MODEL, GROUNDING_PROMPT, TOP_K, MIN_SCORE

# -------- Sample Data --------
FAQ = [
    (1, "Can I get a refund if I cancel within 24 hours?",
     "Yes. If you cancel within 24 hours of purchase and booked at least 7 days before departure, you'll receive a full refund. After 24 hours, fare rules apply.",
     "24-hour refund"),
    (2, "What if the airline changes my schedule?",
     "For significant schedule changes (e.g., over 3 hours), you can be rebooked at no extra charge or request a refund for the unused portion.",
     "Schedule change policy"),
    (3, "Are non-refundable fares changeable?",
     "Non-refundable tickets can be changed for a fee plus any fare difference, unless restricted by the fare rules.",
     "Fare rules"),
    (4, "How do I rebook after a cancellation?",
     "Use Manage Trip, the mobile app, or the airport desk. You'll be rebooked on the next available flight with similar routing at no extra cost.",
     "IROPs rebooking"),
    (5, "Are taxes refundable on unused tickets?",
     "Government-imposed taxes and fees are usually refundable on unused tickets; carrier-imposed charges may not be.",
     "Taxes & fees"),
]

# -------- Data & Index --------
def load_data():
    path = "airline_faq.csv"
    if os.path.exists(path):
        df = pd.read_csv(path)
        assert {"id", "question", "answer"}.issubset(df.columns)
        return df
    return pd.DataFrame(FAQ, columns=["id", "question", "answer", "source"])

def build_index(df):
    vectorizer = TfidfVectorizer(stop_words="english")
    corpus = (df["question"].fillna("") + " " + df["answer"].fillna("")).tolist()
    doc_matrix = vectorizer.fit_transform(corpus)
    return vectorizer, doc_matrix

def retrieve(query, vectorizer, doc_matrix, df, k=TOP_K, min_score=MIN_SCORE):
    qv = vectorizer.transform([query])
    sims = cosine_similarity(doc_matrix, qv).ravel()
    top = np.argsort(-sims)[:k]
    results = []
    for i in top:
        if sims[i] >= min_score:
            r = df.iloc[i].to_dict()
            r["_score"] = float(sims[i])
            results.append(r)
    return results

def format_context(passages):
    return "\n\n".join(f"[FAQ:{p['id']}] Q: {p['question']}\nA: {p['answer']}" for p in passages)

# -------- Groq Helpers --------
client = Groq(api_key=GROQ_API_KEY)

def groq_generate(prompt):
    try:
        chat = client.chat.completions.create(
            model=GROQ_MODEL,
            temperature=0.0,
            messages=[{"role": "user", "content": prompt}],
        )
        return chat.choices[0].message.content.strip()
    except Exception:
        return "Sorry, the model service is temporarily unavailable."

SMALLTALK_RE = re.compile(
    r"^(hi|hello|hey|yo|hola|sup|good\s*(morning|evening|afternoon)|"
    r"how\s*are\s*you|thanks|thank\s*you|bye|goodbye)\b",
    re.IGNORECASE,
)

def is_smalltalk(text): return bool(SMALLTALK_RE.match(text.strip()))

def groq_smalltalk(user_text):
    prompt = (
        "You are a friendly airline assistant. "
        "Respond briefly to greetings or thanks. "
        "Mention you can help with refunds, changes, baggage, and schedules.\n\n"
        f"User: {user_text}\nAssistant:"
    )
    return groq_generate(prompt)

def grounded_answer(question, passages):
    if not passages:
        if is_smalltalk(question):
            return groq_smalltalk(question)
        return "I don’t have enough info from the FAQ. Please contact Aurora Skies support."
    ctx = format_context(passages)
    prompt = GROUNDING_PROMPT.format(question=question, context=ctx)
    ans = groq_generate(prompt)
    if not any(f"[FAQ:{p['id']}]" in ans for p in passages):
        best = max(passages, key=lambda x: x["_score"])
        return f"{best['answer']} [FAQ:{best['id']}]"
    return ans
