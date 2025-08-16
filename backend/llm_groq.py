import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.runnables import RunnablePassthrough
from langchain_core.messages import HumanMessage
from langchain_huggingface import HuggingFaceEmbeddings


load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")

llm = ChatGroq(model="openai/gpt-oss-120b", groq_api_key=groq_api_key)


os.environ['HF_TOKEN']=os.getenv("HF_TOKEN")
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

base_dir = os.path.join(os.path.dirname(__file__), "..", "lessons_faiss")
base_dir = os.path.abspath(base_dir) 

store = {}

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

def get_all_messages(x):
    return " ".join(msg.content for msg in x["messages"])

def build_rag_chain(file_name: str): 
    path = os.path.join(base_dir, file_name)
    vectorstore = FAISS.load_local(
        folder_path=path,
        embeddings=embeddings,
        allow_dangerous_deserialization=True)
    retriever = vectorstore.as_retriever()

    print(f"Vectorstore loaded from: {path}")

    prompt = ChatPromptTemplate.from_messages([("system", """
                                                You are a Python programming tutor. Only answer questions if the provided <context> is relevant. 
                                                If the question is not related to the context, say "Sorry, I can't answer questions out of the lessons". But if it's related to introductions, that's fine.
                                                
                                                Also, respond to the query in the **same language** as the user.

                                                Follow these instructions when answering:
                                                1. Start with <think> on a new line.
                                                2. Write your internal reasoning (how to solve the problem) inside <think>...</think>. Your reasoning should be at least 500 characters long.
                                                3. After </think>, provide a clear step-by-step explanation.
                                                4. End with: Answer = 

                                                Example:
                                                <think> The user asks in Indonesian what they will learn today. The context is about Python variable naming conventions, including camel case, Pascal case, case sensitivity, and multi-word variable names</think>

                                                Answer = In today's lesson we’ll focus on naming variables in Python:
                                                        Camel case – write the first word in lower‑case and start each subsequent word with a capital letter, e.g. myVariableName.
                                                        Pascal case – start every word with a capital letter, e.g. MyVariableName.
                                                        Variable names are case‑sensitive – myVar, MyVar, and myvar are three different identifiers.
                                                        Multi‑word variable names should avoid spaces; use either camel case or Pascal case, or separate words with underscores (my_variable_name).
                                                        These conventions help make your code readable and consistent.

                                                <context>
                                                {context}
                                                </context>
                                                """),
                                                    MessagesPlaceholder(variable_name="messages")
                                                ])

    # Ambil context dari pertanyaan terakhir
    rag_chain = (
        RunnablePassthrough.assign(
            context=lambda x: retriever.invoke(get_all_messages(x))
        )
        | prompt
        | llm
    )

    print(f"RAG chain built for: {file_name}")

    rag_with_memory = RunnableWithMessageHistory(
        rag_chain,
        get_session_history,
        input_messages_key="messages"
    )

    return rag_with_memory
