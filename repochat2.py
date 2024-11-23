import os
import streamlit as st
from qdrant_client import QdrantClient
from langchain.vectorstores import Qdrant
from langchain.embeddings import HuggingFaceBgeEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chains import ConversationChain
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
import git  # Ensure GitPython is installed: pip install GitPython

# Load environment variables
load_dotenv()

# Set up Streamlit page
st.set_page_config(page_title="GitHub Repository Query Assistant", page_icon=":books:")
st.title("Chat with GitHub Repositories :books:")

# Define model and embeddings
model_name = "BAAI/bge-large-en"
model_kwargs = {'device': 'cpu'}
encode_kwargs = {'normalize_embeddings': False}
embeddings = HuggingFaceBgeEmbeddings(
    model_name=model_name,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs
)

# Qdrant client configuration
url = "https://d4f75fbf-745a-4016-be67-0501b305d8a7.europe-west3-0.gcp.cloud.qdrant.io"
api_key = "DX0sWzqBHxPbZd3wrpLz53f5Cqo-OIIWDFTYME0LX2aglU--fKhd-Q"
client = QdrantClient(url=url, api_key=api_key, prefer_grpc=False)

# Initialize memory with conversation history
memory = ConversationBufferWindowMemory(k=5)

# Function to clone a GitHub repository
def clone_repo(repo_url, target_folder):
    try:
        git.Repo.clone_from(repo_url, target_folder)
        st.success(f"Repository cloned successfully to {target_folder}")
    except Exception as e:
        st.error(f"Error cloning repository: {e}")

# Function to extract text from files in the cloned repository
def extract_text_from_repo(repo_folder):
    text = ""
    for root, _, files in os.walk(repo_folder):
        for file in files:
            if file.endswith(('.md', '.py', '.txt')):  # You can add more file types as needed
                file_path = os.path.join(root, file)
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    text += f.read() + "\n"  # Add a newline to separate contents
    return text

# Function to ingest the text into Qdrant
def ingest_repo_to_qdrant(repo_folder):
    try:
        text = extract_text_from_repo(repo_folder)
        
        # Create a Document object for the repository content
        documents = [Document(page_content=text, metadata={"source": repo_folder})]

        # Split the documents into chunks
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        texts = text_splitter.split_documents(documents)

        # Ingest the documents into Qdrant vector DB
        qdrant = Qdrant.from_documents(
            texts,
            embeddings,
            url=url,
            api_key=api_key,
            prefer_grpc=False,
            collection_name="repochat"  # Use the 'repochat' collection
        )

        st.success("Repository content ingested successfully!")
        return text

    except Exception as e:
        st.error(f"Error ingesting repository content: {e}")
        return None

# Function to query Qdrant and retrieve context
def query_qdrant(query):
    db = Qdrant(client=client, embeddings=embeddings, collection_name="repochat")
    docs = db.similarity_search_with_score(query=query, k=2)

    retrieved_content = "\n".join([doc.page_content for doc, score in docs])
    return retrieved_content, docs

# Sidebar for GitHub repository input
with st.sidebar:
    st.subheader("Input your GitHub repository URL")
    repo_url = st.text_input("Enter GitHub Repository URL")
    if st.button("Clone and Process Repository"):
        if repo_url:
            with st.spinner("Cloning repository..."):
                repo_folder = "cloned_repo"  # Change this to a dynamic folder name if desired
                clone_repo(repo_url, repo_folder)
                with st.spinner("Ingesting content..."):
                    raw_text = ingest_repo_to_qdrant(repo_folder)
                    st.session_state.raw_text = raw_text
        else:
            st.error("Please enter a valid GitHub repository URL.")

# Query input and processing
if "raw_text" in st.session_state:
    user_query = st.text_input("Ask a question about your repository:")
    if user_query:
        with st.spinner("Generating response..."):
            try:
                retrieved_content, docs = query_qdrant(user_query)

                # Define prompt template
                prompt_template = PromptTemplate(
                    input_variables=["query", "context"],
                    template=(
                        "You are an intelligent assistant highly skilled in analyzing, summarizing, and explaining content from GitHub repositories. "
                        "Your goal is to provide a detailed, accurate, and well-structured response based on the user's query and the content extracted from the repository. "
                        "If the query relates to specific sections or details of the document, focus on those parts, ensuring clarity and relevance. "
                        "Summarize complex information effectively, highlighting key insights or critical points. "
                        "If certain aspects of the query cannot be answered due to missing information, acknowledge this and suggest possible next steps. "
                        "In addition, provide a brief summary of the most important insights from the document to offer further value."
                        "If you don't understood the query or you don't have a proper answer do not provide Gibberish and Rubbish response .Just simply say i cant't provide the solution."
                        "\nContext: {context}\n"
                        "Query: {query}\n\n"
                        "Provide an insightful, clear, and comprehensive response, ensuring you address the query using relevant sections of the repository. "
                        "Break down complex topics where necessary, and offer a concise summary of key points at the end of your response."
                    )
                )

                # Format input using the prompt template
                formatted_input = prompt_template.format(context=retrieved_content, query=user_query)

                # Set up Groq model for response generation
                groq_api_key = os.environ['GROQ_API_KEY']
                model = 'mixtral-8x7b-32768'  

                groq_chat = ChatGroq(groq_api_key=groq_api_key, model_name=model)
                conversation = ConversationChain(llm=groq_chat, memory=memory)

                # Generate response
                response = conversation.run(formatted_input)

                # Display response
                st.write("### Response:")
                st.write(response)

            except Exception as e:
                st.error(f"Error during response generation: {e}")
else:
    st.info("Please enter a GitHub repository URL and process it first.")
