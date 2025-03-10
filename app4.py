import os
import streamlit as st
from qdrant_client import QdrantClient
from langchain.vectorstores import Qdrant
from langchain.embeddings import HuggingFaceBgeEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_groq import ChatGroq
from PyPDF2 import PdfReader
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document

# Load environment variables
load_dotenv()

# Set up Streamlit page
st.set_page_config(page_title="PDF Query Assistant", page_icon=":books:")
st.title("Chat with PDF Documents :books:")

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
# Replace with your Qdrant cluster URL and API key
url = "https://0b5900e0-0130-4a65-a184-fdd7ec5d8c1c.europe-west3-0.gcp.cloud.qdrant.io:6333"
api_key = "S-122Acdsexyw5zIkYqsG3rEOBQ__T1sFcKIMEEo4_R_SFTo5e0RMA"  # Replace with your API key

client = QdrantClient(
    url=url,
    api_key=api_key,
    prefer_grpc=False  # Optional: use gRPC if needed
)

# Initialize memory for conversation history
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# Function to process PDF and extract text
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

# Function to ingest the PDF text into Qdrant
def ingest_pdf_to_qdrant(pdf_docs):
    try:
        documents = []
        for pdf in pdf_docs:
            pdf_reader = PdfReader(pdf)
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text()

            documents.append(Document(page_content=text, metadata={"source": pdf.name}))

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        texts = text_splitter.split_documents(documents)

        # Connect to the specific collection on the Qdrant cluster
        qdrant = Qdrant.from_documents(
            texts,
            embeddings,
            url=url,
            prefer_grpc=False,
            api_key=api_key,
            collection_name="chat-with-pd"  # Collection name provided
        )

        st.success("PDFs ingested successfully!")
        return "\n".join([doc.page_content for doc in documents])

    except Exception as e:
        st.error(f"Error ingesting PDFs: {e}")
        return None

# Function to create Conversational Retrieval Chain
def get_conversational_chain():
    db = Qdrant(client=client, embeddings=embeddings, collection_name="chat-with-pd")
    
    llm = ChatGroq(
        groq_api_key=os.environ['GROQ_API_KEY'],
        model_name='mixtral-8x7b-32768'
    )

    # Create conversational retrieval chain
    conversational_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=db.as_retriever(),
        memory=memory
    )
    return conversational_chain

# Sidebar for PDF uploads
with st.sidebar:
    st.subheader("Upload your PDF documents")
    pdf_docs = st.file_uploader("Upload PDFs", accept_multiple_files=True, type=["pdf"])

    if st.button("Process PDFs"):
        with st.spinner("Processing..."):
            try:
                raw_text = ingest_pdf_to_qdrant(pdf_docs)
                st.session_state.raw_text = raw_text
                # Initialize the conversational chain after processing PDFs
                st.session_state.conversational_chain = get_conversational_chain()
            except Exception as e:
                st.error(f"Error processing PDFs: {e}")

# Query input and processing
if "raw_text" in st.session_state:
    user_query = st.text_input("Ask a question about your documents:")

    if user_query:
        with st.spinner("Generating response..."):
            try:
                # Generate response using conversational chain
                response = st.session_state.conversational_chain(
                    {"question": user_query}
                )

                # Store the chat history in the session state
                if 'chat_history' not in st.session_state:
                    st.session_state.chat_history = []

                # Append the user question and the response to the chat history as a pair
                st.session_state.chat_history.append({"role": "user", "content": user_query})
                st.session_state.chat_history.append({"role": "assistant", "content": response["answer"]})

                # Display conversation history with latest messages on top
                st.write("### Conversation History:")
                history = st.session_state.chat_history
                # Display chat history in reverse order, pairing user query and assistant response
                for i in range(len(history) - 1, -1, -2):
                    user_msg = history[i-1] if i-1 >= 0 else None
                    assistant_msg = history[i] if i >= 0 else None

                    if user_msg and user_msg['role'] == 'user':
                        st.write(f"*You*: {user_msg['content']}")
                    if assistant_msg and assistant_msg['role'] == 'assistant':
                        st.write(f"*Assistant*: {assistant_msg['content']}")

            except Exception as e:
                st.error(f"Error during response generation: {e}")
else:
    st.info("Please upload and process your PDFs first.")
