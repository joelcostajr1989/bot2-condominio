import os
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA

def carregar_pdf(caminho_pdf):
    loader = PyPDFLoader(caminho_pdf)
    return loader.load_and_split()

def criar_ou_carregar_vectorstore(docs, persist_directory):
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-MiniLM-L3-v2")
    if os.path.exists(persist_directory):
        return Chroma(persist_directory=persist_directory, embedding_function=embedding_model)
    return Chroma.from_documents(docs, embedding_model, persist_directory=persist_directory)

st.title("Bot do Condomínio (leve e offline)")

uploaded_file = st.file_uploader("Envie o PDF da Ata ou Manual", type="pdf")
query = st.text_input("Faça uma pergunta sobre o documento")

if uploaded_file and query:
    caminho_temp = os.path.join("temp.pdf")
    with open(caminho_temp, "wb") as f:
        f.write(uploaded_file.read())

    docs = carregar_pdf(caminho_temp)
    db = criar_ou_carregar_vectorstore(docs, persist_directory="db")
    qa_chain = RetrievalQA.from_chain_type(llm=None, retriever=db.as_retriever())
    resultado = qa_chain.run(query)
    st.write("Resposta:", resultado)