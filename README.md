
# ThinkStruct-Search

ThinkStruct-Search is a local semantic search system built with **FAISS** and **SentenceTransformers**.  
It supports embedding documents, building a FAISS index, and running semantic search via API or a simple web interface.

---

## 📂 Project Structure

```

.
├── app/
│   ├── api.py              # API server
│   ├── embed\_index.py      # Build FAISS index from embeddings
│   ├── ingest.py           # Load and preprocess data
│   ├── search\_core.py      # Core search logic
│   ├── smoke\_local.py      # Local search test script
│
├── cache/                  # Stores embeddings, index, and metadata
│   ├── meta.jsonl
│   ├── sample\_units.jsonl
│   ├── units.parquet
│   └── vectors.faiss
│
├── data/                   # Raw input data
├── notes/
│   └── ingest\_report.md
│
├── web/                    # Web interface
├── cli.py                  # CLI entry point
├── run\_api.bat             # Windows batch to start API
├── run\_web.bat             # Windows batch to start web UI
└── .gitignore

````

---

## ⚙️ Requirements

- Python 3.9+
- pip (or conda)
- [FAISS](https://github.com/facebookresearch/faiss)
- [SentenceTransformers](https://www.sbert.net/)
- NumPy

---

## 🚀 Installation

```bash
git clone https://github.com/yourusername/thinkstruct-search.git
cd thinkstruct-search
python -m venv .venv
source .venv/bin/activate   # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
````

---

## 📌 Usage

### **Option 1 — No Cache Files**

If you **do not** have prebuilt cache files, you need to run the ingestion and indexing process.

1️⃣ **Prepare your data**
Place your raw input files (e.g., JSONL, CSV, or Parquet) inside the `data/` folder.

2️⃣ **Ingest data**

```bash
python app/ingest.py
```

3️⃣ **Generate embeddings & FAISS index**

```bash
python app/embed_index.py
```

4️⃣ **Test local search**

```bash
python app/smoke_local.py
```

5️⃣ **Start API server**

```bash
python app/api.py
```

6️⃣ **Run the web UI**

```bash
python run_web.bat
```

---

### **Option 2 — Already Have `cache/`**

If you **already have** a complete `cache/` folder (including `vectors.faiss`, `meta.jsonl`, and `units.parquet`), you can skip ingestion and indexing.

**Required cache structure:**

```
cache/
  ├── meta.jsonl
  ├── sample_units.jsonl   # Optional
  ├── units.parquet
  └── vectors.faiss
```

**Run local search immediately:**

```bash
python app/smoke_local.py
```

**Start API server:**

```bash
python app/api.py
```

**Run the web UI:**

```bash
python run_web.bat
```

⚠ **Note:** The cache must match the embedding model used (`sentence-transformers/all-MiniLM-L6-v2` by default).
If your cache was built with a different model, you must regenerate it.

---

## 🧠 How It Works

1. **Ingestion (`ingest.py`)**
   Reads raw data, extracts text, and stores it in a structured format (`units.parquet`).

2. **Embedding (`embed_index.py`)**
   Uses `SentenceTransformer` to encode text into embeddings, stores them in FAISS.

3. **Search (`search_core.py`)**
   Encodes queries, searches FAISS for nearest vectors, filters results by section or classification.

4. **API (`api.py`)**
   Exposes search as a REST API.

5. **Web UI (`web/`)**
   Provides a frontend interface to interact with the search engine.

---

## 📄 Example Search Code

```python
from app.search_core import Searcher

searcher = Searcher("./cache/vectors.faiss", "./cache/meta.jsonl")
results = searcher.search("wheel speed sensor", topk=5)

for r in results:
    print(f"[{r['score']:.4f}] {r['title']} - {r['snippet']}")
```

---

## 📜 License

MIT License. See `LICENSE` for details.

