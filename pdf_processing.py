import os
import streamlit as st
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from qdrant_client import QdrantClient
from langchain.vectorstores import Qdrant
from langchain.embeddings import HuggingFaceBgeEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain_groq import ChatGroq
from PyPDF2 import PdfReader

# Define model and embeddings for Qdrant
model_name = "BAAI/bge-large-en"
model_kwargs = {'device': 'cpu'}
encode_kwargs = {'normalize_embeddings': False}
embeddings = HuggingFaceBgeEmbeddings(
    model_name=model_name,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs
)

# Qdrant client configuration
qdrant_url_pdf = os.getenv('QDRANT_URL_pdfchat')
qdrant_api_key_pdf = os.getenv('QDRANT_API_KEY_pdfchat')

# Function to ingest the PDF text into Qdrant
def ingest_pdf_to_qdrant(pdf_docs):
    try:
        documents = []
        for pdf in pdf_docs:
            pdf_reader = PdfReader(pdf)
            for page_num, page in enumerate(pdf_reader.pages):
                text = page.extract_text()
                if text:
                    documents.append(Document(page_content=text, metadata={"source": f"{pdf.name} - Page {page_num + 1}"}))
                else:
                    st.error(f"Page {page_num + 1} in {pdf.name} has no extractable text.")

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        texts = text_splitter.split_documents(documents)

        qdrant = Qdrant.from_documents(
            texts,
            embeddings,
            url=qdrant_url_pdf,
            api_key=qdrant_api_key_pdf,
            prefer_grpc=False,
            collection_name="chat-with-pd"
        )

        st.success("PDFs ingested successfully!")
        return "\n".join([doc.page_content for doc in documents])

    except Exception as e:
        st.error(f"Error ingesting PDFs: {e}")
        return None

# Function to create Conversational Retrieval Chain for PDF
def get_conversational_chain_pdf():
    client_pdf = QdrantClient(url=qdrant_url_pdf, api_key=qdrant_api_key_pdf, prefer_grpc=False)
    db = Qdrant(client=client_pdf, embeddings=embeddings, collection_name="chat-with-pd")
    
    llm = ChatGroq(
        groq_api_key=os.getenv('GROQ_API_KEY'),
        model_name='mixtral-8x7b-32768'
    )

    conversational_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=db.as_retriever(),
        memory=st.session_state.memory
    )
    return conversational_chain