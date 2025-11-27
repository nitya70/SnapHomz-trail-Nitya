import streamlit as st
import faiss
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import pipeline

# Page title
st.title("📘 RAG Mini Edition — Snaphomz")
st.subheader("Ask anything from your dataset!")

# Input box
query = st.text_input("Ask a question:")

# Load models
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
llm = pipeline("text2text-generation", model="google/flan-t5-small")

# Load FAISS index and data
index = faiss.read_index("remarks_faiss_index.index")
df = pd.read_csv("normalized_dataset.csv")

# Store remarks + full rows
remarks_list = df["remarks"].astype(str).tolist()
all_rows = df.to_dict("records")

if st.button("Search") and query:
    with st.spinner("Searching… 🔍"):

        # Convert query to embedding
        q_emb = embedding_model.encode([query]).astype("float32")

        # Search FAISS
        D, I = index.search(q_emb, 3)

        # Show matched row numbers
        st.subheader("📌 Matched Row Indexes")
        st.write(I[0])

        # Show top remark matches
        st.subheader("🔎 Top Matches (Remarks)")
        for i in I[0]:
            st.write("•", remarks_list[i])

        # Show full row data
        st.subheader("📋 Full Row Data")
        for i in I[0]:
            st.json(all_rows[i])

        # Build context for LLM
        context = "\n".join([remarks_list[i] for i in I[0]])

        # LLM Prompt
        prompt = f"""Use the context to answer correctly.

Context:
{context}

Question: {query}
Answer:"""

        # Generate answer
        result = llm(prompt, max_length=250)

        # Show Answer
        st.subheader("✅ Answer")
        st.write(result[0]["generated_text"])

