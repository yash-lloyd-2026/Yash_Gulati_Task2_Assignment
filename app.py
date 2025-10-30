import streamlit as st
from config import PAGE_TITLE, PAGE_ICON, LAYOUT
from utils import load_data, build_index, retrieve, grounded_answer

# ----------------------------- Page Config -----------------------------
st.set_page_config(page_title=PAGE_TITLE, page_icon=PAGE_ICON, layout=LAYOUT)

# ----------------------------- Style -----------------------------
st.markdown("""
<style>
[data-testid="stAppViewContainer"] {
  background: radial-gradient(1200px 800px at 50% -10%, #0ea5e915, transparent),
              linear-gradient(180deg, #0b1020 0%, #0b0f1a 100%);
}
[data-testid="stHeader"] { background: transparent; }
footer { visibility: hidden; }
.bubble {
  padding: 0.8rem 1rem;
  border-radius: 14px;
  line-height: 1.5;
  border: 1px solid rgba(255,255,255,0.07);
}
.assistant { background: rgba(14,165,233,0.10); }
.user { background: rgba(255,255,255,0.06); }
</style>
""", unsafe_allow_html=True)

# ----------------------------- Data Setup -----------------------------
df = load_data()
vectorizer, doc_matrix = build_index(df)

# ----------------------------- Chat History -----------------------------
if "history" not in st.session_state:
    st.session_state.history = [{
        "role": "assistant",
        "content": "Hi! Ask me anything about Aurora Skies Airways."
    }]

# Header
st.markdown("<h2 style='margin-top:0'>✈️ Aurora Skies — Support Chat</h2>", unsafe_allow_html=True)
st.caption("Grounded answers using TF-IDF retrieval + Groq LLM (with anti-hallucination).")

# Display conversation
for msg in st.session_state.history:
    avatar = "✈️" if msg["role"] == "assistant" else "👤"
    with st.chat_message(msg["role"], avatar=avatar):
        st.markdown(f"<div class='bubble {msg['role']}'>{msg['content']}</div>", unsafe_allow_html=True)

# ----------------------------- Chat Input -----------------------------
user_input = st.chat_input("Type your question…")

if user_input:
    # 1️⃣ Immediately show user's message
    st.session_state.history.append({"role": "user", "content": user_input})
    with st.chat_message("user", avatar="👤"):
        st.markdown(f"<div class='bubble user'>{user_input}</div>", unsafe_allow_html=True)

    # 2️⃣ Retrieve relevant FAQs
    retrieved = retrieve(user_input, vectorizer, doc_matrix, df)

    # 3️⃣ Generate grounded answer
    reply = grounded_answer(user_input, retrieved)

    # 4️⃣ Show assistant’s reply
    st.session_state.history.append({"role": "assistant", "content": reply})
    with st.chat_message("assistant", avatar="✈️"):
        st.markdown(f"<div class='bubble assistant'>{reply}</div>", unsafe_allow_html=True)

    # 5️⃣ Refresh to display updated conversation correctly
    st.rerun()
