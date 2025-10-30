# ✈️ Aurora Skies – RAG Support Chatbot
This project is a simple, high-speed chatbot for a fictional airline, Aurora Skies, built using Streamlit and the Groq API. It uses a technique called Retrieval-Augmented Generation (RAG) to ensure the answers are factual and based only on a provided set of internal FAQs.

# ✨ Key Features
This application provides grounded responses (answers based only on the FAQ data), uses TF-IDF for fast document retrieval, handles smalltalk gracefully, and delivers real-time chat performance thanks to the Groq LLM.

# 🚀 Setup & Run

1. Requirements
You need Python and a Groq API Key.

2. Installation
Download all project files (app.py, config.py, utils.py, requirements.txt).
Install the necessary libraries:
pip install -r requirements.txt

3. API Key Configuration
Edit the config.py file to replace the placeholder with your actual Groq API key:
config.py
 ... other config ...
GROQ_API_KEY = "YOUR_GROQ_API_KEY_HERE"  # <-- Update this
 ... rest of file ...

4. Run the Application
Start the chatbot from your terminal:
streamlit run app.py


# 🛠️ How It Works
The core of the application is a RAG pipeline:
When a user asks a question, the system first finds the most relevant FAQs using a simple TF-IDF index.
These retrieved FAQ passages are sent to the Groq LLM (llama-3.1-8b-instant) along with a strict prompt to ensure it generates an answer only from the context, including citations (e.g., [FAQ:1]).

# 📁 Files

app.py: The Streamlit UI and chat management.
utils.py: Core RAG logic: data loading, TF-IDF retrieval, and Groq API calls.
config.py: Stores the API key, LLM name, and the grounding prompt.
requirements.txt: List of Python dependencies.
