# Setup & Installation Guide

This guide covers setting up and running the **Software & AI Patent Infringement Auditor** locally or via Docker.

---

## 1. Prerequisites

- **Python**: Version `3.10` or higher
- **Ollama**: Local LLM server ([Download Ollama](https://ollama.com/))
- **Docker Desktop**: For PostgreSQL with `pgvector` ([Download Docker](https://www.docker.com/products/docker-desktop/))

---

## 2. Option A: Docker Compose (All-in-One Setup)

The repository provides a multi-container Docker composition that starts PostgreSQL (`pgvector`), Ollama, and the Streamlit web application simultaneously.

```bash
# 1. Clone the repository
git clone https://github.com/laluni/software-ai-patent-auditor.git
cd software-ai-patent-auditor

# 2. Build and start all services
docker compose up --build -d

# 3. Pull the recommended LLM inside the Ollama container
docker exec -it patent_ollama ollama pull qwen2.5:latest
```

Open your browser at **[http://localhost:8501](http://localhost:8501)**.

---

## 3. Option B: Local Python Environment Setup

### Step 1: Clone Repository & Create Virtual Environment
```bash
git clone https://github.com/laluni/software-ai-patent-auditor.git
cd software-ai-patent-auditor

# Create virtual environment
python -m venv venv

# Activate on Windows (PowerShell)
.\venv\Scripts\Activate.ps1

# Activate on Linux/macOS
source venv/bin/activate
```

### Step 2: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 3: Start Vector Database
```bash
docker compose up -d postgres
```

### Step 4: Pull Ollama Model
```bash
ollama pull qwen2.5:latest
```

### Step 5: Launch Streamlit Web App
```bash
streamlit run app.py
```
Open **[http://localhost:8501](http://localhost:8501)** in your browser.

---

## 4. Running Benchmarks & Tests

### Run Ingestion Pipeline (`dlt`)
```bash
python -m src.dlt_ingest
```

### Run Retrieval Evaluation
```bash
python -m src.eval
```

### Run Automated Unit & Integration Tests
```bash
python -m pytest tests/
```
