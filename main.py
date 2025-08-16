from fastapi import FastAPI
from pydantic import BaseModel
from langchain_core.messages import HumanMessage
from fastapi.middleware.cors import CORSMiddleware
from backend.llm_groq import build_rag_chain  # pastikan path sesuai
import re


def extract_answer_only(response: str) -> str:
    
    pattern = response.split("</think>")[-1]
    pattern = re.sub(r"(?i)\nAnswer\s*=\s*", "", pattern)
    pattern = re.sub(r"(?i)\nJawaban\s*=\s*", "", pattern)
    if pattern:
        return pattern
    
    return response.strip()

app = FastAPI()

# Tambahkan ini untuk mengizinkan frontend (localhost:5173) mengakses backend
app.add_middleware(
    CORSMiddleware, 
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatInput(BaseModel):
    prompt: str
    lesson: str  # nama folder FAISS yang akan dipanggil, misal: "for_loops"

@app.post("/chat")
def chat(input: ChatInput):
    print("✅ Prompt:", input.prompt)
    print("📁 Lesson:", input.lesson)
    # Bangun chain berdasarkan materi
    rag_chain = build_rag_chain(file_name=input.lesson)

    config = {
        "configurable": {"session_id": f"chat-{input.lesson}"}
    }

    # Kirim ke LLM
    response = rag_chain.invoke(
        {
            "messages": [HumanMessage(content=input.prompt)]
        },
        config=config
    )

    bot_response = response.content

    pure_answer = extract_answer_only(bot_response)

    return {"response": pure_answer}
