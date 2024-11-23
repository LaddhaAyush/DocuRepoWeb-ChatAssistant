import os
import streamlit as st
from dotenv import load_dotenv
from pdf_processing import ingest_pdf_to_qdrant, get_conversational_chain_pdf
from repo_processing import clone_repo, ingest_repo_to_qdrant, get_conversational_chain_repo
from image_processing import generate_image
from langchain.memory import ConversationBufferMemory
from langchain.document_loaders import UnstructuredURLLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
import google.generativeai as genai
import groq
import numpy as np
import pandas as pd
import faiss

# Load environment variables
load_dotenv()

# Set up Streamlit page
st.set_page_config(page_title="Document & Repository Query Assistant", page_icon="🌐📚")
st.title("Chat with Web URLs, PDFs, or GitHub Repositories 🌐📚")

# Initialize session state variables
if "raw_text" not in st.session_state:
    st.session_state.raw_text = ""
if "conversational_chain" not in st.session_state:
    st.session_state.conversational_chain = None
if "memory" not in st.session_state:
    st.session_state.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# FAISS-related settings
EMBEDDING_DIM = 768  # Google's embedding dimension
embeddings_file_path = "faiss_embeddings.index"
metadata_file_path = "faiss_metadata.csv"

# Set up Groq client
groq_client = groq.Groq(api_key=os.getenv('GROQ_API_KEY'))

# Set up Google API key for embeddings
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

class FAISSVectorStore:
    def __init__(self, dimension):
        self.dimension = dimension
        self.index = faiss.IndexFlatL2(dimension)
        self.docs = []

    def add_documents(self, documents, embeddings):
        self.index.add(np.array(embeddings).astype('float32'))
        self.docs.extend(documents)

    def save(self):
        faiss.write_index(self.index, embeddings_file_path)
        metadata = pd.DataFrame([
            {"page_content": doc.page_content, "source": doc.metadata["source"]}
            for doc in self.docs
        ])
        metadata.to_csv(metadata_file_path, index=False)

    @classmethod
    def load(cls, dimension):
        if not os.path.exists(embeddings_file_path) or not os.path.exists(metadata_file_path):
            raise FileNotFoundError("FAISS index or metadata file not found.")
        
        index = faiss.read_index(embeddings_file_path)
        metadata = pd.read_csv(metadata_file_path)
        if metadata.empty:
            raise ValueError("Metadata file is empty.")

        store = cls(dimension)
        store.index = index
        store.docs = [
            Document(page_content=row['page_content'], metadata={"source": row['source']})
            for _, row in metadata.iterrows()
        ]
        return store

    def query(self, query_embedding, top_k=3):
        D, I = self.index.search(np.array([query_embedding]).astype('float32'), top_k)
        return [self.docs[i] for i in I[0]] if I.size > 0 else []

# Sidebar for user selection
option = st.sidebar.selectbox("Choose an option:", ["Chat with Web URL", "Chat with PDF", "Chat with GitHub Repository","Generate Image"])
if option == "Generate Image":
        st.subheader("Enter design keywords")
        keywords = st.text_input("Enter design keywords")

        if st.button("Generate Design"):
            if keywords:
                with st.spinner("Generating design..."):
                    image = generate_image(keywords)
                    if image:
                        st.image(image, caption="Generated Interior Design", use_column_width=True)
            else:
                st.error("Please enter design keywords.")
# Main logic for Web URL interaction
elif option == "Chat with Web URL":
    with st.sidebar:
        st.subheader("Enter a Web URL")
        web_url = st.text_input("Web URL")

        if st.button("Process URL"):
            progress_bar = st.progress(0)
            with st.spinner("Processing URL..."):
                try:
                    loader = UnstructuredURLLoader(urls=[web_url])
                    data = loader.load()
                    progress_bar.progress(30)

                    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
                    docs = text_splitter.split_documents(data)
                    progress_bar.progress(50)

                    embeddings = []
                    for doc in docs:
                        embedding_result = genai.embed_content(
                            model="models/embedding-001",
                            content=doc.page_content,
                            task_type="retrieval_document",
                            title="Embedding of document"
                        )
                        embeddings.append(embedding_result["embedding"])

                    progress_bar.progress(70)

                    faiss_store = FAISSVectorStore(dimension=EMBEDDING_DIM)
                    faiss_store.add_documents(docs, embeddings)
                    progress_bar.progress(90)

                    faiss_store.save()
                    progress_bar.progress(100)

                    st.session_state.faiss_store = faiss_store
                    st.session_state.raw_text = "\n".join([doc.page_content for doc in docs])
                    st.success("URL processed successfully!")
                except Exception as e:
                    st.error(f"Error processing URL: {e}")

    # Query input and processing for Web URL
    if "faiss_store" in st.session_state:
        user_query = st.text_input("Ask a question about the processed URL:")

        if user_query:
            with st.spinner("Generating response..."):
                try:
                    query_embedding_result = genai.embed_content(
                        model="models/embedding-001",
                        content=user_query,
                        task_type="retrieval_query"
                    )
                    query_embedding = query_embedding_result["embedding"]

                    sorted_docs = st.session_state.faiss_store.query(query_embedding, top_k=5)

                    if not sorted_docs:
                        st.warning("No relevant data found in the processed URL.")
                    else:
                        combined_text = " ".join([doc.page_content for doc in sorted_docs])

                        chat_completion = groq_client.chat.completions.create(
                            messages=[
                                {"role": "system", "content": f"You are a helpful assistant. Use the following context to answer the user's question: {combined_text}"},
                                {"role": "user", "content": user_query}
                            ],
                            model="mixtral-8x7b-32768",
                        )

                        answer = chat_completion.choices[0].message.content

                        st.session_state.chat_history.append({"role": "user", "content": user_query})
                        st.session_state.chat_history.append({"role": "assistant", "content": answer})

                        st.header("Answer")
                        st.write(answer)

                        st.write("### Conversation History:")
                        for i in range(len(st.session_state.chat_history) - 1, -1, -2):
                            user_msg = st.session_state.chat_history[i-1] if i-1 >= 0 else None
                            assistant_msg = st.session_state.chat_history[i] if i >= 0 else None

                            if user_msg and user_msg['role'] == 'user':
                                st.write(f"You: {user_msg['content']}")
                            if assistant_msg and assistant_msg['role'] == 'assistant':
                                st.write(f"Assistant: {assistant_msg['content']}")

                except Exception as e:
                    st.error(f"Error during response generation: {e}")
    else:
        st.info("Please process a URL first.")

# Main logic for PDF interaction
elif option == "Chat with PDF":
    with st.sidebar:
        st.subheader("Upload your PDF documents")
        pdf_docs = st.file_uploader("Upload PDFs", accept_multiple_files=True, type=["pdf"])

        if st.button("Process PDFs"):
            with st.spinner("Processing..."):
                try:
                    raw_text = ingest_pdf_to_qdrant(pdf_docs)
                    st.session_state.raw_text = raw_text
                    st.session_state.conversational_chain = get_conversational_chain_pdf()
                except Exception as e:
                    st.error(f"Error processing PDFs: {e}")

    # Query input and processing for PDF
    if st.session_state.raw_text:
        user_query = st.text_input("Ask a question about your PDF documents:")

        if user_query:
            with st.spinner("Generating response..."):
                try:
                    # Generate the response
                    response = st.session_state.conversational_chain(
                        {"question": user_query}
                    )

                    # Display response
                    st.write("### Assistant Response:")
                    st.write(response['answer'])

                    # Add the user query and assistant response to the memory
                    st.session_state.memory.chat_memory.add_user_message(user_query)
                    st.session_state.memory.chat_memory.add_ai_message(response['answer'])

                    # Display chat history with user query above the response
                    st.write("### Chat History:")
                    st.write(f"User: {user_query}")
                    st.write(f"Assistant: {response['answer']}")

                    # Display previous messages in reverse order
                    for message in reversed(st.session_state.memory.chat_memory.messages):
                        if message.content not in [user_query, response['answer']]:
                            role = "User" if message.type == "human" else "Assistant"
                            st.write(f"{role}:** {message.content}")

                except Exception as e:
                    st.error(f"Error during response generation: {e}")
    else:
        st.info("Please upload and process your PDFs first.")

# Main logic for GitHub repository interaction
elif option == "Chat with GitHub Repository":
    with st.sidebar:
        st.subheader("Input your GitHub repository URL")
        repo_url = st.text_input("Enter GitHub Repository URL")

        if st.button("Clone and Process Repository"):
            if repo_url:
                with st.spinner("Cloning repository..."):
                    repo_folder = clone_repo(repo_url)
                    if repo_folder:
                        with st.spinner("Ingesting content..."):
                            raw_text = ingest_repo_to_qdrant(repo_folder)
                            st.session_state.raw_text = raw_text
                            st.session_state.conversational_chain = get_conversational_chain_repo()

    # Query input and processing for GitHub repository
    if st.session_state.raw_text:
        user_query = st.text_input("Ask a question about the repository:")

        if user_query:
            with st.spinner("Generating response..."):
                try:
                    # Generate the response
                    response = st.session_state.conversational_chain(
                        {"question": user_query}
                    )

                    # Display response
                    st.write("### Assistant Response:")
                    st.write(response['answer'])

                    # Add the user query and assistant response to the memory
                    st.session_state.memory.chat_memory.add_user_message(user_query)
                    st.session_state.memory.chat_memory.add_ai_message(response['answer'])

                    # Display chat history with user query above the response
                    st.write("### Chat History:")
                    st.write(f"User: {user_query}")
                    st.write(f"Assistant: {response['answer']}")

                    # Display previous messages in reverse order
                    for message in reversed(st.session_state.memory.chat_memory.messages):
                        if message.content not in [user_query, response['answer']]:
                            role = "User" if message.type == "human" else "Assistant"
                            st.write(f"{role}:** {message.content}")

                except Exception as e:
                    st.error(f"Error during response generation: {e}")
    else:
        st.info("Please input and process your GitHub repository first.")

# Add a button to clear the chat history
if st.button("Clear Chat History"):
    st.session_state.memory.clear()
    st.session_state.chat_history = []
    st.success("Chat history cleared!")

# Run the Streamlit app
if __name__ == "__main__":
    st.sidebar.markdown("---")
    st.sidebar.markdown("Made by TY-G11")