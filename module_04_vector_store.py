"""
============================================================
 MODULE 4: The Vector Store (ChromaDB)
============================================================

 WHAT WE'RE DOING:
    In Module 3, we split our PDF into chunks. Now we need to:
      1. Convert each chunk into an "Embedding" (a list of numbers)
      2. Store those embeddings in a "Vector Database" (ChromaDB)
      3. Search the database to find chunks similar to a question

 WHAT IS AN EMBEDDING?
    An embedding converts text into a list of numbers (a "vector").
    
    Example (simplified):
      "The cat sat on the mat"  → [0.2, 0.8, 0.1, 0.9, ...]
      "A kitten was on the rug" → [0.2, 0.7, 0.1, 0.8, ...]  ← SIMILAR!
      "Stock market crashed"    → [0.9, 0.1, 0.8, 0.2, ...]  ← DIFFERENT!
    
    Similar meanings → similar numbers → we can find related text!
    
 WHAT IS A VECTOR DATABASE?
    A special database optimized for storing embeddings and finding
    "similar" vectors quickly. Think of it like Google Search but
    for your own private documents.

 WHY HUGGINGFACE EMBEDDINGS?
    They're FREE and run LOCALLY on your computer.
    No API key needed! (Unlike OpenAI embeddings which cost money)

⚠️ THE PROBLEM:
    If we use ChromaDB's default mode, the database lives only in
    RAM (memory). Every time you restart your script → data is GONE!
    You'd have to re-process the entire PDF again. 

 THE SOLUTION:
    Use a "Persistent" directory. ChromaDB saves the data to your
    hard drive, so it survives restarts. Like saving a game! 

▶️ HOW TO RUN:
    python module_04_vector_store.py
    
    NOTE: You need a PDF file. Update `pdf_path` below.
    First run will download the embedding model (~90MB) — one-time only.
"""

# ============================================================
# STEP 1: Imports
# ============================================================

import os
import warnings

# Suppress warnings and telemetry logs BEFORE importing Chroma
warnings.filterwarnings("ignore", category=FutureWarning)
os.environ["ANONYMIZED_TELEMETRY"] = "False"
os.environ["CHROMA_TELEMETRY"] = "False"

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# HuggingFaceEmbeddings — free, local embedding model
# It converts text into vectors (lists of numbers)
from langchain_huggingface import HuggingFaceEmbeddings

# Chroma — our vector database
# We use the LangChain wrapper which makes it super easy to use
from langchain_chroma import Chroma


# ============================================================
# STEP 2: Load and Chunk the PDF (same as Module 3)
# ============================================================

pdf_path = "TransformerAttenctionMechanism.pdf"  #  Change to YOUR PDF path

print(f" Loading PDF: {pdf_path}")
loader = PyPDFLoader(pdf_path)
pages = loader.load()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)
chunks = text_splitter.split_documents(pages)
print(f"✂️  Created {len(chunks)} chunks from {len(pages)} pages\n")


# ============================================================
# STEP 3: Create the Embedding Model
# ============================================================

# We're using "all-MiniLM-L6-v2" — a small but powerful model
# It converts any text into a 384-dimensional vector
# (384 numbers that represent the "meaning" of the text)
#
#  FIRST RUN: This will download the model (~90MB)
#    After that, it runs from your local cache — no internet needed!

print(" Loading embedding model (first time downloads ~90MB)...")

embeddings = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2"
)

# Let's see what an embedding looks like!
sample_embedding = embeddings.embed_query("Hello world")
print(f" Embedding dimensions: {len(sample_embedding)}")
print(f" First 5 values: {sample_embedding[:5]}")
print(f"   (These numbers represent the 'meaning' of 'Hello world')\n")


# ============================================================
# STEP 4: Create the ChromaDB Vector Store (PERSISTENT!)
# ============================================================

#  KEY CONCEPT: persist_directory
#   - This tells ChromaDB to save data to the "chroma_db" folder
#   - When you restart the script, data is still there!
#   - Without this, everything would be lost on restart

persist_directory = "./chroma_db"

print(" Storing chunks in ChromaDB (persistent)...")

# This does 3 things in one call:
#   1. Takes each chunk's text
#   2. Converts it to an embedding (using our HuggingFace model)
#   3. Stores both the text AND embedding in ChromaDB
vector_store = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory=persist_directory,
    collection_name="pdf_collection"
)

print(f" Stored {len(chunks)} chunks in ChromaDB!")
print(f" Data saved to: {os.path.abspath(persist_directory)}")
print(f"   (This data survives script restarts!)\n")


# ============================================================
# STEP 5: Search the Vector Store!
# ============================================================

# Now the magic — we can search our PDF using natural language!
# ChromaDB converts our question into an embedding and finds
# the chunks with the most similar embeddings.

query = "What is attention mechanism?"

print(f" Searching for: \"{query}\"\n")

# similarity_search returns the most relevant chunks
# k=3 means "give me the top 3 most similar chunks"
results = vector_store.similarity_search(query, k=3)

for i, doc in enumerate(results, 1):
    print(f" Result {i} (Page {doc.metadata.get('page', '?')}):")
    print(f"   {doc.page_content[:200]}...")
    print()


# ============================================================
# BONUS: Loading an EXISTING database (no re-processing!)
# ============================================================

# Next time you run the script, you don't need to re-process the PDF!
# Just load the existing database:

print("="*50)
print(" BONUS: Loading existing database (no re-processing)...\n")

existing_store = Chroma(
    persist_directory=persist_directory,
    embedding_function=embeddings,
    collection_name="pdf_collection"
)

# Search the existing database
results2 = existing_store.similarity_search("Attention", k=1)
if results2:
    print(f" Found from existing DB (Page {results2[0].metadata.get('page', '?')}):")
    print(f"   {results2[0].page_content[:200]}...")

print("\n Vector store created and searchable!")
print(" Next: Module 5 — Building the complete RAG chain")

