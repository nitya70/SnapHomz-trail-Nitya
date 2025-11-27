# Snaphomz RAG Mini Project

This project implements a Retrieval-Augmented Generation (RAG) system over real-estate listings.  
The app allows users to ask natural language questions and retrieves relevant property listings using semantic search.

## Features

- Data cleaning and preprocessing
- Embedding generation using SentenceTransformers
- FAISS vector index for fast similarity search
- Local open-source LLM for answer generation
- Streamlit web interface for interactive Q&A

## Tech Stack

- Python
- Streamlit
- FAISS
- SentenceTransformers
- HuggingFace Transformers
- Pandas, NumPy

## How to Run

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Run the App:

```bash
streamlit run app.py
```

3. Open in browser

```bash
http://localhost:8501
```

How it Works
1. User asks a question
2. Query is converted to embedding
3. FAISS retrieves top matching rows
4. Retrieved context is passed to model
5. The model generates a human-like answer

Limitations
1. Uses a small LLM model due to resource limits
2. Works best for short factual queries
3. Answers depend on quality of remarks data
