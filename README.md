RAG Document Q&A — Multi-PDF Traditional RAG + GraphRAG-Inspired Search

A Streamlit application that answers questions over multiple PDF documents using four retrieval modes: Traditional RAG, Local Search, Global Search, and Hybrid Search.

«Project Overview: This project combines vector-based retrieval with knowledge graph-based retrieval to provide contextual and evidence-backed answers from uploaded documents.

Search Modes: Traditional RAG uses semantic vector retrieval, Local Search focuses on relevant entities and relationships, Global Search uses community-level information, and Hybrid Search combines retrieval approaches.

LLM Integration: Google Gemini is used for language generation and supported extraction tasks.

Caching: Persistent LLM response caching is implemented using SQLite to reuse previously generated responses.

Deployment: The application is containerized using Docker and Docker Compose.»

---

1. Objective

Develop an intelligent document question-answering system that allows users to upload PDF documents, process their contents, and ask natural-language questions.

The system supports four retrieval approaches to provide both focused and broader contextual answers.

The primary objectives are:

- Extract and process information from PDF documents.
- Generate semantic embeddings using Sentence Transformers.
- Store and retrieve document embeddings using ChromaDB.
- Construct a knowledge graph using extracted entities and relationships.
- Organize related entities into communities.
- Support Traditional RAG, Local Search, Global Search, and Hybrid Search.
- Generate evidence-backed responses with citations.
- Reduce repeated LLM API calls through persistent caching.

2. Architecture

The application follows a modular architecture consisting of document ingestion, embedding generation, vector storage, knowledge graph construction, retrieval, and response generation.

2.1 Document Ingestion

- Upload PDF documents through the Streamlit interface.
- Extract text using PyPDF.
- Split extracted content into manageable text chunks.
- Prepare document chunks and metadata for indexing.

2.2 Vector Processing

- Generate embeddings using Sentence Transformers.
- Store vector representations in ChromaDB.
- Retrieve semantically relevant document chunks based on user queries.

2.3 Knowledge Graph Construction

- Extract entities and relationships from document content.
- Represent entities and their relationships as a graph.
- Use NetworkX for graph representation and processing.
- Apply community detection to organize related entities.
- Generate community reports for broader information retrieval.

2.4 Query Processing

User queries are processed through the selected retrieval mode.

The retrieved information is passed to the language model for contextual answer generation.

2.5 Response Generation

- Integrate Google Gemini API for response generation.
- Use retrieved document context and graph information.
- Provide evidence-backed answers and citations.

3. Retrieval Modes

The application provides four retrieval modes.

3.1 Traditional RAG

Uses vector similarity search to retrieve relevant document chunks from ChromaDB.

The retrieved context is passed to Gemini to generate answers relevant to the user's question.

3.2 Local Search

Uses knowledge graph entities and their relationships to retrieve focused information relevant to a specific query.

It helps identify connections between related entities within the document collection.

3.3 Global Search

Uses knowledge graph communities and generated community reports to retrieve broader information across the document collection.

It is designed for questions that require understanding information from multiple related topics.

3.4 Hybrid Search

Combines multiple retrieval approaches to obtain complementary contextual information.

The retrieved evidence is used to generate a comprehensive response.

4. Persistent LLM Caching

The application implements persistent LLM response caching using SQLite.

The caching mechanism stores previously generated responses and reuses them when matching requests are made.

Key features:

- SQLite-based persistent cache storage.
- Namespace-based cache organization.
- Cache key-based response retrieval.
- Cache HIT and MISS tracking.
- Reuse of previously generated responses.
- Reduced repeated Gemini API calls.

Cache HIT: A matching cached response is available and reused.

Cache MISS: No matching response is available, so a new LLM request is performed and the response can be stored.

This mechanism helps reduce redundant API calls and improve application efficiency.

5. Technology Stack

Programming Language: Python 3.11

Frontend: Streamlit

LLM: Google Gemini API

LLM Framework: LangChain

Embedding Model: Sentence Transformers

Vector Database: ChromaDB

Knowledge Graph: NetworkX

Graph Processing: Graspologic

Community Detection: Leiden Algorithm

PDF Processing: PyPDF

Caching: SQLite

Containerization: Docker and Docker Compose

6. Project Structure

Rag-Document-QA/
│
├── app.py
├── ingestion.py
├── vector_store.py
├── graph_rag.py
├── search.py
├── evidence.py
├── llm_cache.py
│
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
├── .dockerignore
├── .gitignore
├── README.md
│
└── .env.example

«Runtime-generated files such as local databases, uploaded PDFs, vector-store data, and cache files are excluded from version control.»

7. Installation and Execution

7.1 Prerequisites

- Python 3.11
- Git
- Google Gemini API key
- Docker Desktop (optional, for containerized execution)

7.2 Clone the Repository

git clone https://github.com/r-sanjana/Rag-Document-QA.git
cd Rag-Document-QA

7.3 Create a Virtual Environment

python -m venv .venv

Activate the environment on Windows:

.venv\Scripts\activate

7.4 Install Dependencies

python -m pip install --upgrade pip
pip install -r requirements.txt

7.5 Configure Environment Variables

Create a ".env" file in the project root directory.

Add the following:

GEMINI_API_KEY=your_gemini_api_key

«Replace the placeholder with your actual Gemini API key. Ensure the variable name matches the application's configuration. Never commit your ".env" file to GitHub.»

7.6 Run the Application

streamlit run app.py

Open the application:

http://localhost:8501

8. Docker Deployment

The project includes Docker support for containerized execution.

8.1 Build the Docker Image

docker compose build

8.2 Start the Application

docker compose up -d

8.3 Access the Application

http://localhost:8501

8.4 Check Running Containers

docker ps

8.5 View Application Logs

docker compose logs -f

8.6 Stop the Container

docker compose down

«Docker Compose uses a named volume for ChromaDB persistence. SQLite cache persistence across container recreation requires a separate persistent volume configuration.»

9. Environment and Security

- API keys are maintained through environment variables.
- ".env" files are excluded from Git.
- Local databases and generated cache files are excluded from version control.
- Uploaded PDF documents are excluded from the repository.
- Deployment platforms should use their secure secrets or environment-variable configuration.

10. Future Enhancements

- Support for additional document formats.
- Improved retrieval and ranking mechanisms.
- Advanced caching and performance optimization.
- Enhanced document management.
- Improved deployment and persistent storage capabilities.
