好的，那我把这个 README 用纯文本直接发给你，你就可以从这里复制到你的 GitHub 项目里了。

---

# Patent Paragraph Search

## 1. Specific Problem Statement

Patent searching is a critical step in the innovation lifecycle. Patent agents, inventors, and legal professionals often need to quickly locate **specific claims or technical descriptions** across a large set of patent filings.
The challenge is that:

* **Keyword search** can miss relevant results that use different wording.
* **Semantic search** can retrieve conceptually similar results but lacks domain-specific filtering (e.g., CPC classification codes, abstract keywords).

**Goal:**
Build a patent paragraph search tool that:

1. Supports **semantic search** using vector embeddings for better conceptual matching.
2. Allows **hybrid searching** (semantic + keyword) to improve precision and recall.
3. Enables users to filter results by patent section (claims / description) and CPC classification prefix.

---

## 2. How the Code Addresses the Problem

### Architecture Overview

* **Backend:** FastAPI (Python)

  * Embedding model: `sentence-transformers/all-MiniLM-L6-v2`
  * Vector index: **FAISS** for semantic similarity search
  * Keyword index: **BM25** (via `rank_bm25`) for lexical relevance
  * Hybrid Search: Combines BM25 score and FAISS score with a tunable weight
* **Frontend:** Simple HTML/JS interface

  * Input fields for query, TopK results, section filter, CPC prefix
  * Checkbox to enable Hybrid Search and weight slider
* **Data:** Provided JSON patent dataset (`patents_ipa{DATE}.json`)

### Search Modes Implemented

1. **Vector Only (Semantic Search)**

   * Uses FAISS to find top-K most semantically similar paragraphs.
2. **Hybrid Search (BM25 + Vector)**

   * Combines keyword relevance (BM25) and semantic relevance (FAISS).
   * Allows weighting between the two methods (default 0.5 each).

### Enhancements Implemented (Part 2)

**Enhancement:** **Hybrid searching**

* **Why:**
  Patent professionals often require **both** conceptual similarity and exact keyword matches. For example:

  * A CPC-specific filter can quickly narrow results to relevant subclasses.
  * BM25 ensures key technical terms appear in the results.
  * Semantic search retrieves conceptually similar results even if exact terms differ.
* **Impact:**
  Improves precision for niche queries (e.g., `wheel speed sensor` in vehicle patents with CPC code `B60B`) while retaining recall for broader queries.

---

## 3. How to Run the Code

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

## 4. Example Usage

**Query:** `wheel speed sensor for vehicle`

* **Vector Only**: Finds conceptually similar paragraphs, even if "wheel speed sensor" is paraphrased.
* **Hybrid Search (weight=0.5)**: Prioritizes results mentioning "wheel speed sensor" explicitly while considering semantic similarity.

---

## 5. Optional Performance Notes

* Tested on \~1,000 patent records.
* Hybrid Search adds minimal latency (<50ms) over pure vector search.
* CPC prefix filtering significantly reduces search space, improving performance on larger datasets.

---

## 6. File Structure

```
app/
  api.py              # FastAPI server
  embed_index.py      # Build FAISS index from dataset
  search_core.py      # Core search logic (BM25 + FAISS)
  ingest.py           # Data loading & preprocessing
cache/                # Stores vectors.faiss & meta.jsonl
data/                 # Patent JSON files
web/                  # Frontend HTML/JS interface
```

---

## 7. Demo Video

Watch the 2-minute walkthrough here: [Google Drive Link](https://drive.google.com/file/d/1kmxlGWaxJFaCU8cq-uqW8pZCnHLfcVmS/view?usp=sharing)
- Vector Only search
- Hybrid search with different weights
- Optional CPC prefix filter

---

## 8. References

* [FAISS Documentation](https://faiss.ai/)
* [Sentence Transformers](https://www.sbert.net/)
* [BM25 Ranking](https://en.wikipedia.org/wiki/Okapi_BM25)




