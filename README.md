# 🧠 LLM Exercise – Python Tutor Chatbot

This project is an educational chatbot application powered by RAG (Retrieval-Augmented Generation), combining:

* **LLM via Groq (`gemma2-9b-it`)**
* **Embeddings from HuggingFace (`all-MiniLM-L6-v2`)**
* **Streamlit** as an alternative interface (legacy)
* **Markdown files** as the knowledge base

---

## 🗂 Project Structure

```bash
llm-exercise/
│
├── lessons/                   # Raw markdown lesson files
│   ├── python_variables.md
│   └── ...
│
├── lessons_faiss/             # FAISS indexes generated from lessons
│   ├── python_variables/
│   └── ...
│
├── backend/                   # FastAPI backend source
│   ├── llm_groq.py            # RAG chain logic
│   ├── build_faiss_index.py   # Indexes markdown → FAISS
│
├── legacy_streamlit.py        # Streamlit-based fallback UI
├── .env                       # Environment variables
├── requirements.txt           # Python dependencies
```

---

## ⚙️ How It Works

1. **Preprocessing:**

   * Run `build_faiss_index.py` to embed markdown files into FAISS index using `all-MiniLM-L6-v2`.

2. **Backend:**

   * The `/chat` FastAPI endpoint receives `prompt` and `lesson` → builds a contextual RAG chain via `llm_groq.py`.

3. **RAG Chain:**

   * Retrieves relevant chunks via FAISS
   * Combines them with user prompt
   * Uses `ChatGroq` (Groq API) to generate answers in Bahasa Indonesia.

4. **Frontend:**

   * Streamlit UI (`legacy_streamlit.py`) provides a side-by-side learning/chat interface.

---

## 📘 Sample Lesson Files

```bash
lessons/
├── python_variables.md
├── python_list.md
├── if_else.md
├── while.md
├── for_loops.md
└── functions.md
```

---

## 🚀 How to Run

### 1. Generate FAISS Index

```bash
python backend/build_faiss_index.py
```

### 2. Start Frontend (Streamlit)

```bash
streamlit run legacy_streamlit.py
```

---

## 🔐 .env File

Your `.env` file should look like:

```
GROQ_API_KEY=your_groq_api_key_here
HF_TOKEN=your_huggingface_api_key_here
```

---

## 📦 Python Requirements

All packages are listed in `requirements.txt`. Some highlights include:

* `langchain`
* `langchain-groq`
* `langchain-ollama`
* `faiss-cpu`
* `streamlit`
* `ollama`
* `python-dotenv`

---
## ⚠️ Important Notes

* Embedding model must be consistent across indexing and querying (`all-MiniLM-L6-v2`)
* Ensure `lesson` sent from frontend matches the folder name inside `lessons_faiss`
* If you want to use Ollama directly for LLM (instead of Groq), you can swap out the chain logic accordingly
