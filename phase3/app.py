# ✅ Streamlit Web App: PDF RAG Chatbot

import streamlit as st
from pypdf import PdfReader
from io import BytesIO
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from openai import OpenAI
import os
from dotenv import load_dotenv

os.environ["TOKENIZERS_PARALLELISM"] = "false" #disabling parallel processing


# Load .env for OpenRouter API
load_dotenv(dotenv_path=".env")
api_key = os.getenv("OPENROUTER_API_KEY")
client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=api_key)

# Load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Streamlit UI
st.title("📄 Chat with Your PDF (RAG + Pinecone style)")

uploaded_file = st.file_uploader("Upload a PDF", type="pdf")

if uploaded_file:
    reader = PdfReader(uploaded_file)
    raw_text = "\n".join([page.extract_text() for page in reader.pages if page.extract_text()])

    # Split into chunks
    def chunk_text(text, chunk_size=300):
        words = text.split()
        return [" ".join(words[i:i+chunk_size]) for i in range(0, len(words), chunk_size)]

    chunks = chunk_text(raw_text)
    embeddings = model.encode(chunks).astype("float32")

    # Create FAISS index
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)

    # Chat history
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = [
            {"role": "system", "content": "You are a helpful assistant."}
        ]

    # Search top-k relevant chunks
    def search_context(query, top_k=3):
        query_vec = model.encode([query]).astype("float32")
        _, I = index.search(query_vec, top_k)
        return [chunks[i] for i in I[0]]

    # Generate answer
    def generate_answer(user_q):
        context = "\n".join(search_context(user_q))
        prompt = f"""Use the context below to answer the question:

Context:
{context}

Question: {user_q}
Answer:"""
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        response = client.chat.completions.create(
            model="mistralai/mistral-7b-instruct",
            messages=st.session_state.chat_history,
            temperature=0.3
        )
        reply = response.choices[0].message.content
        st.session_state.chat_history.append({"role": "assistant", "content": reply})
        return reply

    # Input box
    user_input = st.text_input("Ask a question about the PDF")
    if user_input:
        with st.spinner("Thinking..."):
            answer = generate_answer(user_input)
            st.markdown(f"**You:** {user_input}")
            st.markdown(f"**Bot:** {answer}")


# echo 'export PATH=$PATH:~/Library/Python/3.9/bin' >> ~/.zshrc
# source ~/.zshrc
