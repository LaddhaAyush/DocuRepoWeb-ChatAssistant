import os
import streamlit as st
import time
import git
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from qdrant_client import QdrantClient
from langchain.vectorstores import Qdrant
from langchain.embeddings import HuggingFaceBgeEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain_groq import ChatGroq

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
qdrant_url_repo = os.getenv('QDRANT_URL_repochat')
qdrant_api_key_repo = os.getenv('QDRANT_API_KEY_repochat')

# Function to clone GitHub repository
def clone_repo(repo_url):
    try:
        repo_name = repo_url.split("/")[-1].replace(".git", "")
        target_folder = f"{repo_name}_{int(time.time())}"
        git.Repo.clone_from(repo_url, target_folder)
        st.success(f"Repository cloned successfully to {target_folder}")
        return target_folder
    except git.exc.GitCommandError as e:
        st.error(f"Invalid repository URL or GitHub rate limit hit: {e}")
        return None
    except Exception as e:
        st.error(f"Error cloning repository: {e}")
        return None

# Function to extract text from GitHub repository
def extract_text_from_repo(repo_folder):
    text = ""
    try:
        for root, _, files in os.walk(repo_folder):
            for file in files:
                if file.endswith(('.md', '.py', '.txt')):
                    file_path = os.path.join(root, file)
                    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                        text += f.read() + "\n"
        return text
    except Exception as e:
        st.error(f"Error extracting text from repository: {e}")
        return None

# Function to ingest GitHub repository content into Qdrant
def ingest_repo_to_qdrant(repo_folder):
    try:
        text = extract_text_from_repo(repo_folder)
        
        documents = [Document(page_content=text, metadata={"source": repo_folder})]

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        texts = text_splitter.split_documents(documents)

        qdrant = Qdrant.from_documents(
            texts,
            embeddings,
            url=qdrant_url_repo,
            api_key=qdrant_api_key_repo,
            prefer_grpc=False,
            collection_name="repochat"
        )

        st.success("Repository content ingested successfully!")
        return text

    except Exception as e:
        st.error(f"Error ingesting repository content: {e}")
        return None

# Function to create Conversational Retrieval Chain for GitHub Repos
def get_conversational_chain_repo():
    client_repo = QdrantClient(url=qdrant_url_repo, api_key=qdrant_api_key_repo, prefer_grpc=False)
    db = Qdrant(client=client_repo, embeddings=embeddings, collection_name="repochat")
    
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