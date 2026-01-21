# Jarvis Assistant (Streamlit + Pinecone)

A self-hosted personal assistant that retrieves enterprise knowledge from Pinecone and answers questions with a local LLM. The UI is a Streamlit chatbot, and the backend uses embeddings + retrieval to ground answers.

## Features
- Streamlit chatbot UI
- Pinecone-based document search
- Local LLM responses (Ollama-compatible)
- Optional chat memory stored in Pinecone
- Simple ingestion pipeline

## Project Workflow
See [PROJECT_WORKFLOW.md](PROJECT_WORKFLOW.md) for the end-to-end flow.

## Requirements
- Python 3.10+
- A running local LLM server (Ollama recommended)
- Pinecone API key and index

## Install & Run (GitHub)
Clone the repository, set up a virtual environment, install dependencies, configure environment variables, ingest data, then run the app.

### 1) Clone the repo
```bash
git clone https://github.com/lokesh0221/Jarvis_Diligent.git
cd Jarvis_Diligent
```

### 2) Create and activate a virtual environment
```bash
python -m venv .venv
.venv\Scripts\activate
```

### 3) Install dependencies
```bash
pip install -r requirements.txt
```

### 4) Configure environment variables
Copy [\.env.example](.env.example) to .env and fill in your keys and settings:

```bash
copy .env.example .env
```

Key variables:
- PINECONE_API_KEY
- PINECONE_INDEX
- PINECONE_CLOUD
- PINECONE_REGION
- PINECONE_NAMESPACE
- CHAT_NAMESPACE
- OLLAMA_HOST
- LLM_MODEL

### 5) Ingest documents
Place .txt or .md files under [data](data) or another folder and run:

```bash
python scripts/ingest.py --path data
```

### 6) Run the Streamlit app
```bash
streamlit run app/streamlit_app.py
```

## Usage Notes
- The Streamlit sidebar lets you adjust LLM model, namespace, and retrieval size.
- If the Pinecone index does not exist, it will be created using PINECONE_CLOUD and PINECONE_REGION.
- Chat memory is stored in CHAT_NAMESPACE and used for responses.

## Troubleshooting
- If ingestion fails, verify Pinecone credentials and index settings in .env.
- If the app cannot respond, verify your local LLM server is running and the model name matches LLM_MODEL.
