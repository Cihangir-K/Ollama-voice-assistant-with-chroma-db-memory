from langchain_community.document_loaders import WebBaseLoader

from langchain_community.document_loaders import PyPDFLoader

from PyPDF2 import PdfReader
import os 

from langchain_community.vectorstores import Chroma
from langchain_community import embeddings
from langchain_community.chat_models import ChatOllama
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain.text_splitter import CharacterTextSplitter
import tkinter as tk
from tkinter import filedialog

def process_input():

    root = tk.Tk()
    root.withdraw()

    file_path = filedialog.askopenfilename()


    print(file_path)
    print()

    # file_paths = [pdf_file.name for pdf_file in pdf_files]

    print("file_paths", file_path)
    # pdfs_list = str(pdfs).split("\n")
    # print("pdfs_list", pdfs_list)

    # docs = [PyPDFLoader(pdfs_list).load() for pdfs in pdfs_list]
    docs = PdfReader(file_path)
    # docs_list = [item for sublist in docs for item in sublist]
    # text_splitter = CharacterTextSplitter.from_tiktoken_encoder(chunk_size=7500, chunk_overlap=100)
    # docs_splits = text_splitter.split_documents(docs_list)

    # extract the text
    if docs is not None:
      pdf_reader = PdfReader(file_path)
      text = ""
      for page in pdf_reader.pages:
        text += page.extract_text()

    # split into chunks
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
      )
    docs_splits = text_splitter.split_text(text)
    docs_splits2 = text_splitter.create_documents(docs_splits)
    print("docs_splits:", docs_splits)

    #2. Convert documents to Embeddings and store them


    vectorestore = Chroma.from_documents(
            documents=docs_splits2,
            collection_name="rag-chroma",
            embedding=embeddings.ollama.OllamaEmbeddings(model="nomic-embed-text"),persist_directory="./chroma_db",
    )
    # retriever = vectorestore.as_retriever()

process_input()