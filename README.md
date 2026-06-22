# Tech Notes Q&A Bot — RAG Demo with LangChain + Claude

A beginner-friendly project that demonstrates **RAG (Retrieval-Augmented Generation)** using LangChain and Anthropic's Claude API.

---

## What is RAG?

Normally, a language model like Claude only knows what it learned during training. It cannot answer questions about your private documents or internal knowledge.

**RAG solves this** by combining two things:
- A **search system** that finds relevant text from your documents
- A **language model** that reads that text and generates a clear answer

```
Your .txt files  →  [INDEXING]   →  ChromaDB (vector database on disk)
                                            ↓
User question    →  [RETRIEVAL]  →  Top 3 relevant text chunks
                                            ↓
                 →  [GENERATION] →  Claude API  →  Final Answer
```

---

## Tech Stack

| Component | Tool | Why |
|---|---|---|
| RAG framework | LangChain | Wires all pieces together |
| LLM (answer generation) | Claude Haiku (Anthropic) | Fast, cheap, great for demos |
| Embeddings (text → vectors) | `sentence-transformers` | Free, runs locally, no API key |
| Vector database | ChromaDB | Simple, file-based, no server needed |
| Env variable management | python-dotenv | Keeps API keys out of code |

---

## Project Structure

```
RaGExperiment/
├── main.py                 # The full RAG pipeline (heavily commented)
├── requirements.txt        # Python dependencies
├── .env                    # Your API key goes here (never commit this)
├── data/
│   ├── nodejs_notes.txt    # Sample Node.js engineering notes
│   └── android_notes.txt   # Sample Android engineering notes
└── chroma_db/              # Auto-created: the vector database on disk
```

---

## How It Works (3-Step Pipeline)

### Step 1: Indexing (runs once)

`index_documents()` in `main.py`

1. **Load** — Reads all `.txt` files from the `data/` folder
2. **Split** — Breaks text into 500-character chunks (with 50-char overlap so sentences aren't cut off)
3. **Embed** — Converts each chunk into a vector (a list of ~384 numbers that represents the chunk's meaning)
4. **Store** — Saves all vectors + original text into ChromaDB on disk

> What is an embedding?
> Think of it like a "meaning fingerprint." Text with similar meaning gets similar vectors.
> This is how the system can find relevant chunks for a question even if the exact words don't match.

### Step 2: Retrieval (runs per question)

When you ask a question:
1. The question is converted to a vector using the same embedding model
2. ChromaDB finds the top-3 stored chunks whose vectors are closest to the question's vector
3. These chunks are the "relevant context"

### Step 3: Generation (runs per question)

1. The retrieved chunks are assembled into a context block
2. Claude receives a prompt: *"Answer this question using ONLY the context below: ..."*
3. Claude reads the context and generates a clear answer

---

## Setup & Run

### Prerequisites
- Python 3.9+
- An Anthropic API key ([get one here](https://console.anthropic.com/))

### 1. Activate the virtual environment

```bash
source .venv/bin/activate
```

Your terminal prompt will change to `(.venv) ...` to confirm it's active.

> This isolates project dependencies from your system Python — same idea as `node_modules/` in Node.js.

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

> First run downloads the embedding model (~90MB). This only happens once.

### 3. Add your API key

Open `.env` and replace the placeholder with your real key:

```
ANTHROPIC_API_KEY=sk-ant-...your-real-key...
```

### 4. Run

```bash
python main.py
```

---

## Example Session

```
============================================================
  Tech Notes Q&A Bot (RAG Demo)
============================================================

Ready! Ask me anything about the documents in the 'data/' folder.

Your question: What is the event loop in Node.js?

--- RETRIEVAL: Searching for relevant chunks ---
    Chunk 1 from: data/nodejs_notes.txt
    Preview: The event loop is the heart of Node.js. It is what allows Node.js...

--- GENERATION: Asking Claude ---

Answer:
The event loop is Node.js's mechanism for handling many connections
simultaneously without creating a new thread for each one. It works by
registering a callback for async operations and moving on immediately,
then calling the callback when the operation completes...
```

---

## Test These Questions

| Question | Expected behavior |
|---|---|
| `What is the event loop in Node.js?` | Answers from `nodejs_notes.txt` |
| `What is an Intent in Android?` | Answers from `android_notes.txt` |
| `What is machine learning?` | Says it doesn't have that information |

The third question is the most important — it proves the bot only uses YOUR documents, not Claude's general training knowledge.

---

## Adding Your Own Documents

1. Drop any `.txt` file into the `data/` folder
2. Delete the `chroma_db/` folder (to force a fresh index)
3. Run `python main.py` again — it will re-index and pick up the new file

---

## Key Concepts Glossary

| Term | Plain English |
|---|---|
| **Embedding** | A list of numbers that represents the "meaning" of a piece of text |
| **Vector database** | A database that stores embeddings and can find similar ones quickly |
| **Similarity search** | Finding stored embeddings that are mathematically closest to a query embedding |
| **Chunk** | A small piece of text (e.g. 500 characters) split from a larger document |
| **Context window** | The maximum amount of text Claude can read in one request |
| **Prompt** | The full text sent to Claude, including instructions + context + question |
