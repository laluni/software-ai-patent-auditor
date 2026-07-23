# Setup & Installation Guide

Follow these steps to run the **AI Patent Prior-Art & Trade-Secret Auditor** locally.

---

## 1. Prerequisites

- **Python**: Version `3.10` or higher
- **Ollama**: Installed and running ([Download Ollama](https://ollama.com/))
- **Docker Desktop**: Installed and running (for PostgreSQL + PGVector)

---

## 2. Environment Setup

### Step A: Clone Repository & Create Virtual Environment
```bash
git clone https://github.com/your-username/ai-patent-auditor.git
cd ai-patent-auditor

# Create virtual environment
python -m venv venv

# Activate on Windows (PowerShell)
.\venv\Scripts\Activate.ps1

# Activate on Linux/macOS
source venv/bin/activate
```

### Step B: Install Dependencies
```bash
pip install -r requirements.txt
```

---

## 3. Launch Services

### Step A: Start Vector Database (Docker Container)
Ensure Docker Desktop is open, then start the PGVector PostgreSQL container:
```bash
docker compose up -d
```

### Step B: Pull Ollama Model
Pull the recommended Qwen 2.5 model:
```bash
ollama pull qwen2.5:latest
```

---

## 4. Run Application & Evaluation

### Run Web Dashboard (Streamlit)
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501`.

### Run Retrieval Evaluation Benchmark
To test retrieval Hit Rate and MRR metrics:
```bash
python -m src.eval
```
