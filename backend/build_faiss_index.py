import os
from dotenv import load_dotenv
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import UnstructuredMarkdownLoader
from langchain_huggingface import HuggingFaceEmbeddings

load_dotenv()

path = os.path.join(os.path.dirname(__file__), "..", "lessons")
path = os.path.abspath(path)

os.environ['HF_TOKEN']=os.getenv("HF_TOKEN")
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

for file in os.listdir(path):
    loader=UnstructuredMarkdownLoader(os.path.join(path, file))
    docs=loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=20)
    list_text = text_splitter.split_documents(docs)

    vectorstore = FAISS.from_documents(list_text, embedding=embeddings)
    output_dir = os.path.join("lessons_faiss", file.split(".")[0])
    vectorstore.save_local(output_dir)

print("Faiss Vectore success")