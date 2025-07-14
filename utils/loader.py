from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma

def carregar_documento(path):
    """Carrega e divide o documento PDF em partes menores (chunks)."""
    loader = PyPDFLoader(path)
    documentos = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    textos = splitter.split_documents(documentos)
    return textos

def criar_base_vetorial(textos, embedding_model):
    """Cria uma base vetorial a partir dos textos divididos."""
    db = Chroma.from_documents(textos, embedding_model)
    return db
