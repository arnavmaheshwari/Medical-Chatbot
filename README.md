# Medical Chatbot

A Retrieval-Augmented Generation (RAG) based conversational AI for medical inquiries.
This project is a web-based chatbot built with Flask, LangChain, Pinecone, and Google Gemini that processes medical PDF documents and answers user questions by retrieving relevant context.

# Features

* **Conversational Interface**: Interactive web-based UI for seamless user querying.
* **Retrieval-Augmented Generation (RAG)**: Extracts precise context from medical texts to ground the LLM's answers.
* **Semantic Search**: Employs Pinecone Vector Database to quickly find the most relevant document chunks.
* **Document Processing Pipeline**: Automatically loads, splits, and embeds large PDF files.
* **AI Integration**: Powered by Google's `gemini-robotics-er-1.5-preview` model for highly accurate, context-aware responses.

# Tech Stack

| Category | Technologies |
| --- | --- |
| **Frontend** | HTML, CSS, Flask Templates |
| **Backend** | Python, Flask |
| **Database** | Pinecone Vector Database |
| **AI/ML** | LangChain, Google Gemini, FastEmbed, HuggingFace Embeddings |
| **Cloud/DevOps** | Docker, GitHub Actions |
| **Other Libraries** | PyPDF, python-dotenv |

# Architecture / Project Structure

The application is structured into modular components separating the web server, vector index initialization, and core RAG logic.

```text
.
├── app.py                  # Main Flask application and API routing
├── store_index.py          # Script for processing PDFs and populating the vector DB
├── src/                    # Core source code
│   ├── helper.py           # Functions for loading PDFs, splitting text, and embeddings
│   └── prompt.py           # System prompts for the LangChain QA implementation
├── data/                   # Directory containing source PDFs (e.g., Medical_book.pdf)
├── templates/              # HTML frontend templates (chat.html)
├── static/                 # Static assets (styles.css)
├── requirements.txt        # Python dependencies
├── Dockerfile              # Docker container configuration
└── .github/workflows/      # CI/CD pipeline definitions

```

# How It Works

1. **Data Ingestion**: The system loads PDF files from the `data/` directory.
2. **Text Processing**: Text is extracted and split into chunks of 500 characters with a 20-character overlap to preserve context across boundaries.
3. **Embedding Generation**: Text chunks are converted into mathematical vectors (embeddings) using a HuggingFace MiniLM/FastEmbed model.
4. **Vector Storage**: These embeddings are stored securely in a Pinecone vector index named `medical-chatbot`.
5. **User Query**: When a user asks a question via the web interface, the prompt is embedded and compared against the database to find the top 3 most relevant context chunks.
6. **Response Generation**: The retrieved context and user query are fed into the Google Gemini LLM, which is instructed to provide a concise and medically grounded answer.

# Installation

### Prerequisites

* Python 3.10+
* Pinecone API Key
* Google Gemini API Key

### Setup Instructions

1. **Clone the repository**
```bash
git clone <repository-url>
cd medical-chatbot

```


2. **Install dependencies**
```bash
pip install -r requirements.txt

```


3. **Set up environment variables**
Create a `.env` file in the root directory (see the Environment Variables section for required keys).
4. **Populate the Vector Database**
Ensure your PDF files are located in the `data/` folder, then run:
```bash
python store_index.py

```


5. **Run locally**
```bash
python app.py

```


The application will start on `http://0.0.0.0:8080`.

# Environment Variables

| Variable Name | Description | Required / Optional |
| --- | --- | --- |
| `PINECONE_API_KEY` | API key to authenticate with the Pinecone Vector Database. | Required |
| `GEMINI_API_KEY` | API key to access Google's Gemini LLM. | Required |

# Usage

1. Ensure your `.env` file is properly configured.
2. Run `store_index.py` at least once to index your document data.
3. Start the application using `python app.py`.
4. Open your web browser and navigate to the local host address.
5. Use the chat window to ask medical questions and receive AI-generated answers grounded in your documents.

# API Documentation

| Base URL | Endpoint | Method | Description |
| --- | --- | --- | --- |
| `http://0.0.0.0:8080` | `/` | GET | Renders the primary HTML chat interface. |
| `http://0.0.0.0:8080` | `/get` | POST | Accepts form data containing the user message (`msg`), executes the RAG pipeline, and returns the LLM's response. |

# Database

* **Database Technology**: Pinecone Vector Database
* **Index Name**: `medical-chatbot`
* **Vector Dimension**: 384
* **Distance Metric**: Cosine Similarity
* **Cloud Environment**: AWS (`us-east-1`)

# Screenshots

# Deployment

This project is containerized using Docker and utilizes GitHub Actions for continuous integration.

* **Docker**: The application uses a `python:3.10-slim` base image. The Dockerfile installs dependencies, copies the source code, and exposes port 8080.
* **GitHub Actions**: A CI/CD pipeline (`ci-cd.yml`) is configured to build the Docker image automatically upon pushing to the `master` branch. It executes a smoke test by spinning up a detached container and verifying it runs successfully.

# Testing

Testing is handled automatically via the CI/CD pipeline. The GitHub Actions workflow builds the application image and runs the container locally on an Ubuntu runner to ensure there are no startup failures or runtime dependency issues.

# Performance / Optimization

* **Optimized Chunking**: The `RecursiveCharacterTextSplitter` uses a tuned `chunk_size` of 500 and `chunk_overlap` of 20 to balance semantic completeness with token limit constraints.
* **Thread Management**: The LangChain Pinecone VectorStore is explicitly configured with `pool_threads=1` to ensure stable concurrent connections.
* **Fast Embeddings**: Leverages `BAAI/bge-small-en-v1.5` via FastEmbed for highly performant, local embedding generation before interacting with Pinecone.

# Security

* **Secrets Management**: Sensitive API credentials are never hardcoded. They are managed securely using the `python-dotenv` package and loaded from local `.env` files.
* **No Cache Installs**: The Docker build process utilizes `--no-cache-dir` to prevent stale packages and reduce the container's attack surface.

# Future Improvements

* **Conversational Memory**: Integrate LangChain's memory modules to allow the chatbot to handle multi-turn conversations and follow-up questions.
* **Dynamic File Uploads**: Update the web interface to allow users to upload new PDFs directly through the browser instead of relying on the local `data/` folder.
* **Unit Testing**: Add comprehensive unit tests for the core logic inside the `src/` directory.

# Contributing

1. Fork the repository.
2. Create a feature branch: `git checkout -b feature/your-feature-name`.
3. Commit your changes: `git commit -m "Add some feature"`.
4. Push to the branch: `git push origin feature/your-feature-name`.
5. Open a Pull Request.

# License

This project is licensed under the Apache License, Version 2.0.

# Author

**Arnav Maheshwari**
Database Engineer

# Acknowledgements

* [LangChain](https://langchain.com/) for simplifying the LLM orchestration.
* [Pinecone](https://www.pinecone.io/) for the serverless vector database.
* [Google Gemini](https://deepmind.google/technologies/gemini/) for providing state-of-the-art conversational AI models.
