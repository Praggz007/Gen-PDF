"""
============================================================
🧠 MODULE 3: PDF Loading & The "Chunking" Problem
============================================================

📖 WHAT WE'RE DOING:
    We need to feed a PDF document to our AI. But there's a catch:
    AI models have a "Token Limit" — they can only read a certain
    amount of text at once (like a person who can only read one
    page at a time).

    So we need to:
      1. LOAD the PDF (extract all the text)
      2. SPLIT it into small "chunks" (bite-sized pieces)

📖 WHAT IS A TOKEN?
    A token is roughly 3/4 of a word. So:
      - "Hello world" = 2 tokens
      - A 100-page PDF = ~75,000 tokens
      - Llama 3's limit = ~8,000 tokens per request
    
    If we send all 75,000 tokens at once → CRASH! 💥

⚠️ THE PROBLEM:
    Loading a 100-page book and sending it all at once to the LLM.
    The LLM hits its "Token Limit" and either crashes or ignores
    most of the content.

✅ THE SOLUTION:
    Text Splitting! We break the document into small chunks.
    
    KEY CONCEPT — "Overlap":
    Imagine splitting a sentence at the wrong place:
      Chunk 1: "The patient was diagnosed with"
      Chunk 2: "diabetes and prescribed insulin"
    
    Without overlap, we lose the connection! With overlap:
      Chunk 1: "The patient was diagnosed with diabetes"
      Chunk 2: "diagnosed with diabetes and prescribed insulin"
    
    Now both chunks have the full context! 🎯

▶️ HOW TO RUN:
    python module_03_pdf_loader.py
    
    NOTE: You need a PDF file to test this. Place any PDF in the
    same folder and update the `pdf_path` variable below.
"""

# ============================================================
# STEP 1: Imports
# ============================================================

# PyPDFLoader reads PDF files and extracts text page by page
from langchain_community.document_loaders import PyPDFLoader

# RecursiveCharacterTextSplitter is the smartest text splitter
# It tries to split at natural boundaries:
#   1. First tries to split at paragraphs (\n\n)
#   2. Then at sentences (\n)
#   3. Then at words (" ")
#   4. Only as a last resort, at characters
from langchain_text_splitters import RecursiveCharacterTextSplitter


# ============================================================
# STEP 2: Load a PDF File
# ============================================================

# 📌 CHANGE THIS to the path of YOUR PDF file
pdf_path = "TransformerAttenctionMechanism.pdf"

print(f"📄 Loading PDF: {pdf_path}\n")

# The loader reads the PDF and creates a list of "Document" objects
# Each Document = one page of the PDF
loader = PyPDFLoader(pdf_path)
pages = loader.load()

# Let's see what we got
print(f"📊 Total pages loaded: {len(pages)}")
print(f"📊 Type of each page: {type(pages[0])}")
print(f"\n📖 Preview of Page 1 (first 500 characters):")
print("-" * 50)
print(pages[0].page_content[:500])
print("-" * 50)

# Each Document object has two parts:
#   .page_content = the actual text
#   .metadata = info like page number, source file, etc.
print(f"\n📋 Metadata of Page 1: {pages[0].metadata}")


# ============================================================
# STEP 3: The Problem — Too Much Text!
# ============================================================

# Let's see how much text we have in total
total_characters = sum(len(page.page_content) for page in pages)
estimated_tokens = total_characters // 4  # rough estimate: 1 token ≈ 4 chars

print(f"\n⚠️  Total characters in PDF: {total_characters:,}")
print(f"⚠️  Estimated tokens: {estimated_tokens:,}")
print(f"⚠️  LLM token limit: ~8,000")

if estimated_tokens > 8000:
    print(f"💥 This PDF has {estimated_tokens:,} tokens — WAY over the limit!")
    print(f"🔧 Solution: We need to SPLIT it into smaller chunks.\n")
else:
    print(f"✅ This PDF is small enough, but we'll still chunk it for best results.\n")


# ============================================================
# STEP 4: Split the Text into Chunks
# ============================================================

# Create the text splitter with these settings:
#   chunk_size=1000    → each chunk will be ~1000 characters (~250 tokens)
#   chunk_overlap=200  → last 200 chars of Chunk N = first 200 chars of Chunk N+1
#
# WHY 1000 characters?
#   - Small enough to fit in the LLM's context window
#   - Big enough to contain meaningful information
#
# WHY 200 overlap?
#   - Prevents losing context at chunk boundaries
#   - About 20% overlap is a good rule of thumb

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    length_function=len,         # how to measure chunk size
    is_separator_regex=False     # treat separators as plain strings
)

# Split ALL pages into chunks
chunks = text_splitter.split_documents(pages)

print(f"✂️  Original pages: {len(pages)}")
print(f"✂️  After splitting: {len(chunks)} chunks")
print(f"✂️  Chunk size: ~1000 characters each")
print(f"✂️  Overlap: 200 characters between chunks")


# ============================================================
# STEP 5: Inspect a Chunk
# ============================================================

print(f"\n📦 Preview of Chunk 1:")
print("-" * 50)
print(chunks[0].page_content)
print("-" * 50)
print(f"📋 Chunk 1 metadata: {chunks[0].metadata}")
print(f"📏 Chunk 1 length: {len(chunks[0].page_content)} characters")

# If we have at least 2 chunks, let's show the overlap!
if len(chunks) >= 2:
    print(f"\n🔗 OVERLAP DEMO:")
    print(f"   End of Chunk 1:   ...{chunks[0].page_content[-100:]}")
    print(f"   Start of Chunk 2: {chunks[1].page_content[:100]}...")
    print(f"   👆 Notice how the text overlaps! This preserves context.")

print("\n✅ PDF loaded and chunked successfully!")
print("🎓 Next: Module 4 — Converting chunks into embeddings and storing in ChromaDB")
