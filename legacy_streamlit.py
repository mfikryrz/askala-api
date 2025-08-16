import streamlit as st
import os
import re
from backend.llm_groq import build_rag_chain
from langchain_core.messages import HumanMessage


def extract_answer_only(response: str) -> str:
    
    pattern = response.split("</think>")[-1]
    pattern = re.sub(r"(?i)\nAnswer\s*=\s*", "", pattern)
    pattern = re.sub(r"(?i)\nJawaban\s*=\s*", "", pattern)
    if pattern:
        return pattern
    
    return response.strip()


def format_reward(completions):

    pattern = re.compile(r"^<think>([\s\S]+?)</think>([\s\S]+)")
    rewards = []

    for response in completions:
        match = re.match(pattern, response)
        if match:
            rewards.append(1)
        else:
            rewards.append(0)

    return rewards


# Folder berisi file markdown materi
LESSONS_FOLDER = "lessons"

ordered_lesson = [
    'python_variables',
    'python_list',
    'if_else',
    'while',
    'for_loops',
    'functions',
]

materi_list = []

for filename in ordered_lesson:
    filepath = os.path.join(LESSONS_FOLDER, f"{filename}.md")
    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()

        materi_list.append({
            "file": filename,
            "konten": content
        })

# ==========================
# 2. Session State Setup
# ==========================
if "materi_index" not in st.session_state:
    st.session_state.materi_index = 0

if "chat" not in st.session_state:
    st.session_state.chat = []

# ==========================
# 3. Fungsi Ganti Topik
# ==========================
def next_topic():
    st.session_state.chat = []  # Hapus chat
    st.session_state.materi_index += 1
    if st.session_state.materi_index >= len(materi_list):
        st.session_state.materi_index = 0  # Loop ke awal
    
    # Tambahan penting
    st.session_state["user_input"] = ""  # Reset input

# ==========================
# 4. Layout Dua Kolom
# ==========================
col1, col2 = st.columns([1, 1])
# ==== 📘 Kolom Kiri: Materi ====
with col1:
    st.header("📘 Materi Python")

    with st.container():
        # Tambahkan gaya CSS agar kontainer bisa scroll
        st.markdown(
            """
            <style>
            .scrollable-box {
                width: 100%;             /* buat lebar mengikuti kolom */
                max-height: 600px;           /* opsional: perbesar tinggi juga */
                overflow-y: auto;
                border: 1px solid #ccc;
                border-radius: 8px;
                padding: 1.5em;
                background-color: #f9f9f9;
                font-size: 16px;
            }
            </style>
            """,
            unsafe_allow_html=True
        )

        current = materi_list[st.session_state.materi_index]

        st.markdown(f'<div class="scrollable-box">\n\n{current["konten"]}</div>', unsafe_allow_html=True)

    st.button("➡️ Next Topic", on_click=next_topic)


# ==== 🤖 Kolom Kanan: Chatbot ====
with col2:
    st.header("🤖 Tanya AI")

    user_input = st.text_input("Tanyakan sesuatu:", key="user_input")

    if user_input:
        current = materi_list[st.session_state.materi_index]
        filename = current["file"]

        # Bangun ulang RAG Chain sesuai materi
        rag_chain = build_rag_chain(filename)   

        config = {"configurable": {"session_id": f"chat-{filename}"}}

        response = rag_chain.invoke(
            {"messages": [HumanMessage(content=user_input)]},
            config=config
        )

        bot_response = response.content

        pure_answer = extract_answer_only(bot_response)

        st.session_state.chat.append((user_input, pure_answer))

    for user, bot in st.session_state.chat:
        st.markdown(f"**👤:** {user}")
        st.markdown(f"**🤖:** {bot}")
        st.markdown("---")


        # st.markdown(f"🧠 Reasoning Score: `{score}`")


         # === Evaluasi reward ===
        # rewards = format_reward([bot_response])
        # score = rewards[0]

        # 3. Jika format tidak sesuai, kirim feedback dan invoke ulang
        # if score == 0:
        #     feedback_msg = (
        #         "Please repeat your answer and follow these instructions when answering::\n"
        #         "1. Start with <think> on a new line.\n"
        #         "2. Write your internal reasoning (how to solve the problem) inside <think>...</think>. Your reasoning should be at least 500 characters long.\n"
        #         "3. After </think>, provide a clear step-by-step explanation.\n"
        #         "4. End with: Answer = "
        #     )

        #     response_retry = rag_chain.invoke(
        #         {"messages": [
        #             HumanMessage(content=user_input),
        #             HumanMessage(content=feedback_msg)
        #         ]},
        #         config=config
        #     )

        #     bot_response = response_retry.content
            # rewards = format_reward([bot_response])
            # score = rewards[0]