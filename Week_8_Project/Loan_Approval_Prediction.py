# -------------------------------------------
# Streamlit App: RAG Q&A Chatbot for Loan Approval
# -------------------------------------------

# Install required packages first (if not installed)
# pip install streamlit pandas sentence-transformers transformers

import streamlit as st
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import pipeline

# -------------------------------------------
# Step 1: Load Dataset and Preprocess
# -------------------------------------------

@st.cache_resource
def load_data():
    df = pd.read_csv("Training Dataset.csv")
    # Combine all columns into a single string for retrieval
    df["Combined"] = df.apply(lambda row: " ".join([f"{col}: {str(row[col])}" for col in df.columns]), axis=1)
    return df

@st.cache_resource
def load_models():
    # Load Sentence-BERT and LLM only once
    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    generator = pipeline("text-generation", model="distilgpt2")
    return embedder, generator

@st.cache_resource
def create_embeddings(df, embedder):
    corpus_embeddings = embedder.encode(df["Combined"].tolist(), convert_to_tensor=True)
    return corpus_embeddings

# -------------------------------------------
# Step 2: Define RAG Chatbot Function
# -------------------------------------------

def rag_chatbot(user_question, embedder, generator, df, corpus_embeddings, top_k=3):
    # Embed user question
    question_embedding = embedder.encode(user_question, convert_to_tensor=True)

    # Compute similarity
    similarities = cosine_similarity([question_embedding], corpus_embeddings)[0]

    # Get top_k most relevant rows
    top_indices = np.argsort(similarities)[::-1][:top_k]
    retrieved_docs = df.iloc[top_indices]["Combined"].values

    # Combine retrieved docs as context
    context = "\n".join(retrieved_docs)

    # Create prompt for LLM
    prompt = f"Context:\n{context}\n\nQuestion: {user_question}\nAnswer:"

    # Generate response
    response = generator(prompt, max_length=200, num_return_sequences=1)[0]["generated_text"]

    return response.strip()

# -------------------------------------------
# Step 3: Build Streamlit App Interface
# -------------------------------------------

# App title and description
st.title("🤖 Loan Approval RAG Chatbot")
st.write(
    """
    This chatbot uses **Retrieval-Augmented Generation (RAG)** to answer questions about loan approvals
    based on historical loan data. It retrieves relevant information from the dataset and uses a
    generative AI model to craft intelligent responses.
    """
)

# Load dataset and models
st.info("Loading dataset and models...")
df = load_data()
embedder, generator = load_models()
corpus_embeddings = create_embeddings(df, embedder)
st.success("Models loaded successfully! ✅")

# User input box
user_question = st.text_input("💬 Ask a question about loan approvals:")

if st.button("Get Answer") and user_question.strip() != "":
    with st.spinner("Generating answer..."):
        answer = rag_chatbot(user_question, embedder, generator, df, corpus_embeddings)
        st.markdown(f"### 🤖 Chatbot Answer:")
        st.write(answer)

# Show dataset (optional)
with st.expander("📄 View Dataset"):
    st.dataframe(df.head())

# Footer
st.markdown("---")
st.markdown("✅ **Built with Streamlit, Sentence-BERT, and Hugging Face Transformers**")

