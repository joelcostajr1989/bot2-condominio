import os
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA

# Função para carregar o PDF selecionado na pasta data/
def carregar_pdf(caminho_pdf):
    loader = PyPDFLoader(caminho_pdf)
    return loader.load_and_split()

# Função para criar ou carregar a base de vetores
def criar_ou_carregar_vectorstore(docs, persist_directory):
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-MiniLM-L3-v2")
    if os.path.exists(persist_directory):
        return Chroma(persist_directory=persist_directory, embedding_function=embedding_model)
    return Chroma.from_documents(docs, embedding_model, persist_directory=persist_directory)

st.title("Bot do Condomínio (leve e offline)")

# Lista os PDFs da pasta data/
pdf_files = [f for f in os.listdir("data") if f.endswith(".pdf")]
pdf_selecionado = st.selectbox("Escolha o documento para consulta:", pdf_files)

query = st.text_input("Faça uma pergunta sobre o documento")

if pdf_selecionado and query:
    caminho_pdf = os.path.join("data", pdf_selecionado)
    docs = carregar_pdf(caminho_pdf)
    db = criar_ou_carregar_vectorstore(docs, persist_directory="db")
    qa_chain = RetrievalQA.from_chain_type(llm=None, retriever=db.as_retriever())
    resultado = qa_chain.run(query)
    st.write("Resposta:", resultado)
