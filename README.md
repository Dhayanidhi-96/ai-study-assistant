# 📚 AI Study Assistant

> Chat with any PDF using RAG + LLaMA 3.1 — 100% free to run.

![Python](https://img.shields.io/badge/Python-3.11-blue)
![LangChain](https://img.shields.io/badge/LangChain-0.2-green)
![Groq](https://img.shields.io/badge/Groq-LLaMA3.1-orange)
![Streamlit](https://img.shields.io/badge/Streamlit-1.35-red)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

## 🎯 What It Does

Upload any PDF — textbook, research paper, notes — and ask questions in plain English. The app uses Retrieval-Augmented Generation (RAG) to find the most relevant sections and answer accurately with source references.

## 🏗️ Architecture
```
PDF → PyPDF Loader → Text Splitter → HuggingFace Embeddings
                                           ↓
User Question → History-Aware Retriever → FAISS Vector Store
                        ↓
              LLaMA 3.1 8B (via Groq) → Answer + Sources
```

## 🚀 Quick Start

### 1. Clone the repo
```bash
git clone https://github.com/Dhayanidhi-96/ai-study-assistant
cd ai-study-assistant
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Set up environment
```bash
cp .env.example .env
# Add your Groq API key to .env
```

Get your free Groq API key at: https://console.groq.com

### 4. Run locally
```bash
streamlit run app.py
```

## 🐳 Run with Docker
```bash
# Build and run
docker-compose up --build

# Stop
docker-compose down
```

Open http://localhost:7860

## ☁️ Deploy on Hugging Face Spaces (Free)

1. Go to [huggingface.co/spaces](https://huggingface.co/spaces)
2. Click **Create new Space**
3. Settings:
   - **Name:** ai-study-assistant
   - **SDK:** Docker
   - **Visibility:** Public
4. Push your code:
```bash
git remote add space https://huggingface.co/spaces/YOUR_USERNAME/ai-study-assistant
git push space main
```
5. Go to **Settings → Repository Secrets** → Add `GROQ_API_KEY`
6. Your app is live! 🎉

## ⚙️ Configuration

All settings are in `config.py` — no need to touch core logic:

| Setting | Default | Description |
|--------|---------|-------------|
| `llm_model` | `llama-3.1-8b-instant` | Groq model to use |
| `llm_temperature` | `0.3` | Response creativity |
| `chunk_size` | `1000` | PDF chunk size |
| `retriever_k` | `4` | Chunks retrieved per query |
| `max_file_size_mb` | `10` | Max PDF upload size |

## 🛠️ Tech Stack

| Tool | Purpose |
|------|---------|
| LangChain | RAG pipeline & chains |
| Groq API | Fast LLM inference (free tier) |
| LLaMA 3.1 8B | Language model |
| HuggingFace Embeddings | Text embeddings (free, local) |
| FAISS | Vector similarity search |
| Streamlit | Web UI |
| Docker | Containerization |

## 👤 Author

**Dhayanidhi P** — Final Year Data Science & AI Student  
[GitHub](https://github.com/Dhayanidhi-96) · [LinkedIn](https://www.linkedin.com/in/dhayanidhi-p-3372b0291)

## 📄 License

MIT License — free to use, modify and distribute.