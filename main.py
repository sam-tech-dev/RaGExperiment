# ==============================================================================
# RAG (Retrieval-Augmented Generation) Demo with LangChain + Claude
# ==============================================================================
#
# What is RAG?
# ------------
# Normally, a language model (like Claude) only knows what it learned during
# training. It cannot answer questions about YOUR private documents.
#
# RAG solves this by:
#   1. INDEXING   → Turn your documents into a searchable database
#   2. RETRIEVAL  → When a question arrives, find the most relevant text chunks
#   3. GENERATION → Give those chunks + the question to Claude as context
#
# The flow looks like this:
#
#   Your .txt files  →  [INDEXING]  →  ChromaDB (vector database on disk)
#                                              ↓
#   User question    →  [RETRIEVAL] →  Top 3 relevant text chunks
#                                              ↓
#                    →  [GENERATION] →  Claude API  →  Answer
#
# ==============================================================================

import os
from dotenv import load_dotenv

# LangChain modules — each handles one part of the pipeline
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_anthropic import ChatAnthropic
from langchain.schema import HumanMessage, SystemMessage

# ==============================================================================
# SETUP
# ==============================================================================

# Load environment variables from the .env file
# This is how we safely read the ANTHROPIC_API_KEY without hardcoding it
load_dotenv()

# Directory where we store our text documents
DATA_DIR = "data"

# Directory where ChromaDB will save the vector database on disk
# This is auto-created when we first run indexing
CHROMA_DB_DIR = "chroma_db"

# We'll use this embedding model to convert text → vectors (lists of numbers)
# It runs LOCALLY on your machine — no API key, no cost, no internet needed.
# "all-MiniLM-L6-v2" is small (~90MB), fast, and good enough for demos.
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

# The Claude model to use for generating answers.
# Haiku is the fastest and cheapest Claude model — perfect for learning/testing.
CLAUDE_MODEL = "claude-haiku-4-5-20251001"


# ==============================================================================
# STEP 1: INDEXING
# ==============================================================================
# This function reads your .txt files, splits them into chunks, converts each
# chunk to a vector embedding, and stores everything in ChromaDB.
#
# You only need to run this ONCE (or when you add new documents).
# The database is saved to disk in CHROMA_DB_DIR.
# ==============================================================================

def index_documents():
    print("\n--- INDEXING: Loading documents from the 'data/' folder ---")

    # --- Load documents ---
    # DirectoryLoader scans all .txt files in the data/ folder.
    # TextLoader reads each file as a LangChain "Document" object (text + metadata).
    loader = DirectoryLoader(
        DATA_DIR,
        glob="**/*.txt",       # Match all .txt files, including subfolders
        loader_cls=TextLoader  # Use TextLoader to read plain text files
    )
    documents = loader.load()
    print(f"    Loaded {len(documents)} document(s).")

    # --- Split into chunks ---
    # Why split? Because:
    # 1. Embedding models have a max input size (e.g., 512 tokens)
    # 2. Retrieving a tiny relevant paragraph is better than retrieving a whole book
    # 3. Smaller chunks = more precise search results
    #
    # chunk_size=500   → each chunk is at most 500 characters
    # chunk_overlap=50 → neighboring chunks share 50 characters, so no sentence
    #                    gets brutally cut off at a chunk boundary
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    chunks = splitter.split_documents(documents)
    print(f"    Split into {len(chunks)} chunk(s).")

    # --- Create embedding model ---
    # An "embedding" converts text into a list of ~384 numbers (a vector).
    # Similar-meaning texts get similar vectors — this is how semantic search works.
    #
    # Analogy for engineers: imagine a hash function, but instead of producing
    # a collision-resistant ID, it produces a "meaning fingerprint". Texts with
    # similar meaning have similar fingerprints, so we can measure distance.
    print("    Loading local embedding model (downloads ~90MB on first run)...")
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)

    # --- Store in ChromaDB ---
    # Chroma.from_documents() does three things:
    #   1. Calls embeddings.embed_documents() on each chunk → gets vectors
    #   2. Stores the vectors + original text in the database
    #   3. Saves everything to disk at CHROMA_DB_DIR
    print("    Embedding chunks and saving to ChromaDB...")
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_DB_DIR
    )

    print(f"    Done! Database saved to '{CHROMA_DB_DIR}/'")
    return vectorstore


# ==============================================================================
# STEP 2 + 3: RETRIEVAL + GENERATION
# ==============================================================================
# Given a user question:
#   - RETRIEVAL: Find the top-k most relevant chunks from ChromaDB
#   - GENERATION: Send those chunks + the question to Claude, get an answer
# ==============================================================================

def ask_question(question: str, vectorstore: Chroma) -> str:
    print(f"\n--- RETRIEVAL: Searching for relevant chunks ---")

    # --- Retrieve relevant chunks ---
    # similarity_search() does:
    #   1. Converts the question to a vector (same embedding model used during indexing)
    #   2. Computes "cosine similarity" between the question vector and all stored vectors
    #   3. Returns the top-k most similar chunks
    #
    # Why k=3? For a simple demo, 3 chunks gives enough context without overwhelming the prompt.
    k = 3
    relevant_chunks = vectorstore.similarity_search(question, k=k)

    if not relevant_chunks:
        return "I couldn't find any relevant information in the documents."

    # Show what was retrieved (useful for learning/debugging)
    for i, chunk in enumerate(relevant_chunks):
        source = chunk.metadata.get("source", "unknown")
        print(f"    Chunk {i+1} from: {source}")
        print(f"    Preview: {chunk.page_content[:100]}...")

    # --- Build the context string ---
    # We join the retrieved chunks into one block of text.
    # This block becomes the "context" we provide to Claude.
    context = "\n\n---\n\n".join([chunk.page_content for chunk in relevant_chunks])

    print(f"\n--- GENERATION: Asking Claude ---")

    # --- Set up the Claude LLM ---
    # ChatAnthropic reads ANTHROPIC_API_KEY from the environment automatically.
    llm = ChatAnthropic(model=CLAUDE_MODEL)

    # --- Build the prompt ---
    # This is the core of RAG: we give Claude the retrieved context AND the question.
    # The system message instructs Claude to ONLY use the provided context,
    # which forces it to answer from YOUR documents, not its training data.
    system_prompt = (
        "You are a helpful assistant. Answer the user's question using ONLY the "
        "context provided below. If the answer is not in the context, say "
        "'I don't have information about that in my documents.'\n\n"
        f"CONTEXT:\n{context}"
    )

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=question)
    ]

    # --- Call Claude ---
    response = llm.invoke(messages)

    # response.content is the text answer from Claude
    return response.content


# ==============================================================================
# MAIN
# ==============================================================================

def main():
    print("=" * 60)
    print("  Tech Notes Q&A Bot (RAG Demo)")
    print("=" * 60)

    # --- Check API key is set ---
    # We load it from .env — make sure you updated .env with your real key!
    if not os.getenv("ANTHROPIC_API_KEY") or os.getenv("ANTHROPIC_API_KEY") == "your_api_key_here":
        print("\nERROR: Please set your ANTHROPIC_API_KEY in the .env file.")
        print("  1. Open .env")
        print("  2. Replace 'your_api_key_here' with your real API key")
        print("  3. Get a key at: https://console.anthropic.com/")
        return

    # --- Indexing ---
    # Check if the database already exists on disk.
    # If it does, we skip re-indexing (no need to re-process files every run).
    if os.path.exists(CHROMA_DB_DIR) and os.listdir(CHROMA_DB_DIR):
        print("\nFound existing ChromaDB database. Loading it...")
        embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
        vectorstore = Chroma(
            persist_directory=CHROMA_DB_DIR,
            embedding_function=embeddings
        )
        print("Database loaded.")
    else:
        # First run: build the index from scratch
        vectorstore = index_documents()

    # --- Q&A Loop ---
    # Keep asking questions until the user types 'quit'
    print("\nReady! Ask me anything about the documents in the 'data/' folder.")
    print("Try: 'What is the event loop in Node.js?'")
    print("Try: 'What is an Intent in Android?'")
    print("Try: 'What is machine learning?'  (not in the docs — should say so)")
    print("Type 'quit' to exit.\n")

    while True:
        question = input("Your question: ").strip()

        if not question:
            continue

        if question.lower() in ("quit", "exit", "q"):
            print("Goodbye!")
            break

        answer = ask_question(question, vectorstore)
        print(f"\nAnswer:\n{answer}\n")
        print("-" * 60)


if __name__ == "__main__":
    main()
