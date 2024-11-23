# import os
# import streamlit as st
# import numpy as np
# import pandas as pd
# import faiss
# from langchain.schema import Document
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain.document_loaders import UnstructuredURLLoader
# from dotenv import load_dotenv
# import google.generativeai as genai
# import groq

# # Load environment variables
# load_dotenv()

# # Set up Streamlit page
# st.set_page_config(page_title="Chat with Web URLs", page_icon="🌐")
# st.title("Chat with Web URLs 🌐")

# # FAISS-related settings
# EMBEDDING_DIM = 768  # Google's embedding dimension
# embeddings_file_path = "faiss_embeddings.index"
# metadata_file_path = "faiss_metadata.csv"

# # Set up Groq client
# groq_client = groq.Groq(api_key=os.getenv('GROQ_API_KEY'))

# # Set up Google API key for embeddings
# genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# class FAISSVectorStore:
#     def __init__(self, dimension):
#         self.dimension = dimension
#         self.index = faiss.IndexFlatL2(dimension)
#         self.docs = []

#     def add_documents(self, documents, embeddings):
#         self.index.add(np.array(embeddings).astype('float32'))
#         self.docs.extend(documents)

#     def save(self):
#         faiss.write_index(self.index, embeddings_file_path)
#         metadata = pd.DataFrame([
#             {"page_content": doc.page_content, "source": doc.metadata["source"]}
#             for doc in self.docs
#         ])
#         metadata.to_csv(metadata_file_path, index=False)

#     @classmethod
#     def load(cls, dimension):
#         if not os.path.exists(embeddings_file_path) or not os.path.exists(metadata_file_path):
#             raise FileNotFoundError("FAISS index or metadata file not found.")
        
#         index = faiss.read_index(embeddings_file_path)
#         metadata = pd.read_csv(metadata_file_path)
#         if metadata.empty:
#             raise ValueError("Metadata file is empty.")

#         store = cls(dimension)
#         store.index = index
#         store.docs = [
#             Document(page_content=row['page_content'], metadata={"source": row['source']})
#             for _, row in metadata.iterrows()
#         ]
#         return store

#     def query(self, query_embedding, top_k=3):
#         D, I = self.index.search(np.array([query_embedding]).astype('float32'), top_k)
#         return [self.docs[i] for i in I[0]] if I.size > 0 else []

# # Initialize chat history
# if 'chat_history' not in st.session_state:
#     st.session_state.chat_history = []

# # Sidebar for web URL input
# with st.sidebar:
#     st.subheader("Enter a Web URL")
#     web_url = st.text_input("Web URL")

#     if st.button("Process URL"):
#         with st.spinner("Processing URL..."):
#             try:
#                 loader = UnstructuredURLLoader(urls=[web_url])
#                 data = loader.load()

#                 text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
#                 docs = text_splitter.split_documents(data)

#                 embeddings = []
#                 for doc in docs:
#                     embedding_result = genai.embed_content(
#                         model="models/embedding-001",
#                         content=doc.page_content,
#                         task_type="retrieval_document",
#                         title="Embedding of document"
#                     )
#                     embeddings.append(embedding_result["embedding"])

#                 faiss_store = FAISSVectorStore(dimension=EMBEDDING_DIM)
#                 faiss_store.add_documents(docs, embeddings)
#                 faiss_store.save()

#                 st.session_state.faiss_store = faiss_store
#                 st.success("URL processed successfully!")
#             except Exception as e:
#                 st.error(f"Error processing URL: {e}")

# # Query input and processing
# if "faiss_store" in st.session_state:
#     user_query = st.text_input("Ask a question about the processed URL:")

#     if user_query:
#         with st.spinner("Generating response..."):
#             try:
#                 query_embedding_result = genai.embed_content(
#                     model="models/embedding-001",
#                     content=user_query,
#                     task_type="retrieval_query"
#                 )
#                 query_embedding = query_embedding_result["embedding"]

#                 sorted_docs = st.session_state.faiss_store.query(query_embedding, top_k=5)

#                 if not sorted_docs:
#                     st.warning("No relevant data found in the processed URL.")
#                 else:
#                     combined_text = " ".join([doc.page_content for doc in sorted_docs])

#                     # Using Groq client directly
#                     chat_completion = groq_client.chat.completions.create(
#                         messages=[
#                             {"role": "system", "content": f"You are a helpful assistant. Use the following context to answer the user's question: {combined_text}"},
#                             {"role": "user", "content": user_query}
#                         ],
#                         model="mixtral-8x7b-32768",
#                     )

#                     answer = chat_completion.choices[0].message.content

#                     # Append the user question and the response to the chat history
#                     st.session_state.chat_history.append({"role": "user", "content": user_query})
#                     st.session_state.chat_history.append({"role": "assistant", "content": answer})

#                     st.header("Answer")
#                     st.write(answer)

#                     # st.subheader("Sources:")
#                     # for doc in sorted_docs:
#                     #     st.write(doc.metadata["source"])

#                     # Display conversation history with latest messages on top
#                     st.write("### Conversation History:")
#                     for i in range(len(st.session_state.chat_history) - 1, -1, -2):
#                         user_msg = st.session_state.chat_history[i-1] if i-1 >= 0 else None
#                         # if user_msg:
#                         #  st.markdown(f"<p style='font-size:20px;'>{user_msg}</p>", unsafe_allow_html=True)
#                         assistant_msg = st.session_state.chat_history[i] if i >= 0 else None

#                         if user_msg and user_msg['role'] == 'user':
#                             st.write(f"**You**: {user_msg['content']}")
#                         if assistant_msg and assistant_msg['role'] == 'assistant':
#                             st.write(f"**Assistant**: {assistant_msg['content']}")

#             except Exception as e:
#                 st.error(f"Error during response generation: {e}")
# else:
#     st.info("Please process a URL first.")

import os
import streamlit as st
import numpy as np
import pandas as pd
import faiss
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import UnstructuredURLLoader
from dotenv import load_dotenv
import google.generativeai as genai
import groq
import time  # Import time module for simulating progress

# Load environment variables
load_dotenv()

# Set up Streamlit page
st.set_page_config(page_title="Chat with Web URLs", page_icon="🌐")
st.title("Chat with Web URLs 🌐")

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
        metadata = pd.DataFrame([{"page_content": doc.page_content, "source": doc.metadata["source"]} for doc in self.docs])
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
        store.docs = [Document(page_content=row['page_content'], metadata={"source": row['source']}) for _, row in metadata.iterrows()]
        return store

    def query(self, query_embedding, top_k=3):
        D, I = self.index.search(np.array([query_embedding]).astype('float32'), top_k)
        return [self.docs[i] for i in I[0]] if I.size > 0 else []

# Initialize chat history
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

# Sidebar for web URL input
with st.sidebar:
    st.subheader("Enter a Web URL")
    web_url = st.text_input("Web URL")

    if st.button("Process URL"):
        progress_bar = st.progress(0)  # Initialize progress bar

        with st.spinner("Processing URL..."):
            try:
                loader = UnstructuredURLLoader(urls=[web_url])
                data = loader.load()
                progress_bar.progress(30)  # Update progress

                text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
                docs = text_splitter.split_documents(data)
                progress_bar.progress(50)  # Update progress

                embeddings = []
                for doc in docs:
                    embedding_result = genai.embed_content(
                        model="models/embedding-001",
                        content=doc.page_content,
                        task_type="retrieval_document",
                        title="Embedding of document"
                    )
                    embeddings.append(embedding_result["embedding"])

                progress_bar.progress(70)  # Update progress

                faiss_store = FAISSVectorStore(dimension=EMBEDDING_DIM)
                faiss_store.add_documents(docs, embeddings)
                progress_bar.progress(90)  # Update progress

                faiss_store.save()
                progress_bar.progress(100)  # Final update

                st.session_state.faiss_store = faiss_store
                st.success("URL processed successfully!")
            except Exception as e:
                st.error(f"Error processing URL: {e}")

# Query input and processing
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

                    # Using Groq client directly
                    chat_completion = groq_client.chat.completions.create(
                        messages=[
                            {"role": "system", "content": f"You are a helpful assistant. Use the following context to answer the user's question: {combined_text}"},
                            {"role": "user", "content": user_query}
                        ],
                        model="mixtral-8x7b-32768",
                    )

                    answer = chat_completion.choices[0].message.content

                    # Append the user question and the response to the chat history
                    st.session_state.chat_history.append({"role": "user", "content": user_query})
                    st.session_state.chat_history.append({"role": "assistant", "content": answer})

                    st.header("Answer")
                    st.write(answer)

                    # Display conversation history with latest messages on top
                    st.write("### Conversation History:")
                    for i in range(len(st.session_state.chat_history) - 1, -1, -2):
                        user_msg = st.session_state.chat_history[i-1] if i-1 >= 0 else None
                        assistant_msg = st.session_state.chat_history[i] if i >= 0 else None

                        if user_msg and user_msg['role'] == 'user':
                            st.write(f"**You**: {user_msg['content']}")
                        if assistant_msg and assistant_msg['role'] == 'assistant':
                            st.write(f"**Assistant**: {assistant_msg['content']}")

            except Exception as e:
                st.error(f"Error during response generation: {e}")
else:
    st.info("Please process a URL first.")
