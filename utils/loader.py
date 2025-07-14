from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os

def carregar_documentos(diretorio="data/"):
    documentos = []

    for nome_arquivo in os.listdir(diretorio):
        if nome_arquivo.lower().endswith(".pdf"):
            caminho = os.path.join(diretorio, nome_arquivo)
            loader = PyPDFLoader(caminho)
            documentos.extend(loader.load())

    return documentos

def dividir_documentos(documentos):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100,
        separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""]
    )
    return splitter.split_documents(documentos)
