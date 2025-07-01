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
# import requests
# try:
#     res = requests.get("https://openrouter.ai")
#     st.success(f"🌐 Internet: {res.status_code} - {res.reason}")
# except Exception as e:
#     st.error(f"❌ No internet access or DNS error: {e}")

# # Streamlit UI
# st.title("📄 Chat with Your PDF (RAG + Pinecone style)")
# st.markdown(
#     """<h1 style='text-align: center;'>📄 Chat with Your PDF<br> (RAG + Pinecone style)</h1>""",unsafe_allow_html=True
# )

# st.markdown(
#     """
#     <div style="text-align: center; padding: 1rem 0;">
#         <h1 style="font-size: 2.5rem;">📄 Chat with Your PDF</h1>
#         <h3 style="color: gray; font-weight: normal;">(RAG + Pinecone style)</h3>
#     </div>
#     """,
#     unsafe_allow_html=True
# )

st.markdown(
    """
    <h1 style="
        background: linear-gradient(to right, red, orange, orange, red);
        -webkit-background-clip: text;
        color: transparent;
        font-size: 3rem;
        text-align: center;
        padding: 1rem 0;
    ">
        📄 Chat with Your PDF
    </h1>
    
    """,
    unsafe_allow_html=True
)
# st.success(f"")
# <h3 style="text-align:center; color: white; font-weight: normal;">
#         (RAG + Pinecone style)
#     </h3>


st.markdown("""
    <style>
    div.stButton {
        background: linear-gradient(to right, red, orange, orange, red);
        color: white;
        border: none;
        padding: 0.6rem 1.2rem;
        font-weight: bold;
        border-radius: 12px;
        transition: 0.2s;
        margin-top: 10px;
    }
    div.stButton > button:hover {
        transform: scale(1.03);
        cursor: pointer;
    }
    </style>
""", unsafe_allow_html=True)


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
    

        
    with st.form(key="chat_form", clear_on_submit=True):
        user_input = st.text_input("Ask a question about the PDF", disabled=not True)
        submit = st.form_submit_button("📤 Send", use_container_width=True)

    # Custom container with full CSS styling
    # with st.container():
    #     st.markdown('<div class="custom-box">', unsafe_allow_html=True)
    #     user_input = st.text_input("Ask a question about the PDF")
    #     submit = st.button("📤 Send", key="submit_button", use_container_width=True)
    #     st.markdown('</div>', unsafe_allow_html=True)



    if submit and user_input:
        with st.spinner("Thinking..."):
            answer = generate_answer(user_input)
            # Inject CSS for chat bubbles
            
            # Style chat with custom CSS
            st.markdown("""
            <style>
                .user-msg {
                    background: linear-gradient(to right, #ffe259, #ffa751);  /* Yellow-orange gradient */
                    color: #000;
                    padding: 12px 16px;
                    border-radius: 12px;
                    margin: 12px 0;
                    font-weight: 500;
                }
            
        """, unsafe_allow_html=True)

        # Show chat messages with left-right separation
        st.markdown(f'User: <div class="user-msg">{user_input}</div>', unsafe_allow_html=True)
        st.markdown(f'Bot: <div class="bot-msg">{answer}</div>', unsafe_allow_html=True)



# echo 'export PATH=$PATH:~/Library/Python/3.9/bin' >> ~/.zshrc
# source ~/.zshrc
