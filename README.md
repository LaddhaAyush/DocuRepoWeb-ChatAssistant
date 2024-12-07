
# Chat Assistant for Web URLs, PDFs, and GitHub Repositories

This project is a document and repository query assistant that enables interaction with content from web URLs, PDFs, and GitHub repositories. It leverages Streamlit, LangChain, FAISS, and Qdrant to provide seamless conversational AI capabilities. Users can upload documents or provide URLs/repositories, and the application generates context-aware responses to user queries.


## Features

- Chat with Web URLs: Extract and query content from web pages.
- Chat with PDFs: Upload and query PDF documents.
- Chat with GitHub Repositories: Clone repositories, process their content, and query them conversationally.
- Conversational Memory: Retain chat history for contextual responses.


## Installation

Clone the repository

```bash
 git clone https://github.com/your-repo-name/chat-assistant.git
cd chat-assistant
```
Setup virtual enviroment

```bash
python -m venv env
source env/bin/activate  # On Windows, use `env\Scripts\activate`
```
Install the required Libraries

```bash
pip install -r requirements.txt
```

## Environment Variables

To run this project, you will need to add the following environment variables to your .env file

`GROQ_API_KEY=your_groq_api_key`  
`GOOGLE_API_KEY=your_google_api_key`  
`QDRANT_URL_pdfchat=your_qdrant_url_pdf`   
`QDRANT_API_KEY_pdfchat=your_qdrant_api_key_pdf `  
`QDRANT_URL_repochat=your_qdrant_url_repo `  
`QDRANT_API_KEY_repochat=your_qdrant_api_key_repo `


## Running the Application

- Start the streamlit app
 Exexute the following command:
```bash
streamlit run main.py
```
-  Access the App
 The app will open in your default browser at:
```bash
http://localhost:8501  
```

## How to Use
1 Chat with Web Urls
- Select "Chat with Web URL" from the sidebar.
- Enter a valid web URL and click Process URL.
- Ask questions based on the content extracted from the URL.

2 Chat with PDFs
- Select "Chat with PDF" from the sidebar.
- Upload one or more PDF files and click Process PDFs.
- Query the uploaded documents for relevant information.

3 Chat with GitHub Repositories
- Select "Chat with GitHub Repository" from the sidebar.
- Provide a GitHub repository URL and click Clone and Process Repository.
- Ask questions about the repository content.
## Contributing

Contributions are always welcome!

- Fork the repository.
- Create a feature branch.
- Submit a pull request with a detailed description.


## Acknowledgements

 - LangChain for powerful document processing and retrieval.
- Streamlit for a clean and intuitive user interface.
- Qdrant for robust vector storage and retrieval.
- Google Generative AI for embeddings and model interaction.

