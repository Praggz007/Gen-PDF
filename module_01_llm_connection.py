"""
============================================================
🧠 MODULE 1: Connecting to the LLM (Groq + LangChain)
============================================================

📖 WHAT WE'RE DOING:
    Making our FIRST ever call to an AI model (Large Language Model).
    Think of it like making your first phone call — you need:
      1. A phone number (API Key)
      2. A phone (LangChain framework)
      3. Someone to call (Groq's LLM)

📖 WHAT IS GROQ?
    Groq is a company that lets you use powerful AI models (like Llama 3)
    for FREE with blazing fast speed. We just need an API key from them.

📖 WHAT IS LANGCHAIN?
    LangChain is a Python framework that makes it super easy to build
    apps with LLMs. Instead of writing complex HTTP requests, you just
    write a few lines of Python. It's like a "toolkit" for AI apps.

⚠️ THE PROBLEM THIS MODULE SOLVES:
    If you hardcode your API key like this:
        api_key = "gsk_abc123xyz..."
    And push it to GitHub — ANYONE can steal it and use your account!
    
✅ THE SOLUTION:
    We use `python-dotenv` to load our key from a `.env` file.
    The `.env` file is listed in `.gitignore`, so it never gets pushed.

▶️ HOW TO RUN:
    python module_01_llm_connection.py
"""

# ============================================================
# STEP 1: Import the tools we need
# ============================================================

# `dotenv` loads our secret API key from the .env file
# Without this, we'd have to hardcode the key (BAD practice!)
from dotenv import load_dotenv

# `ChatGroq` is LangChain's connector to Groq's API
# It handles all the HTTP requests, JSON parsing, etc. for us
from langchain_groq import ChatGroq

# `HumanMessage` represents a message from the user
# (There are also SystemMessage, AIMessage — we'll use those in Module 2)
from langchain_core.messages import HumanMessage


# ============================================================
# STEP 2: Load environment variables from .env file
# ============================================================

# This reads the .env file and loads GROQ_API_KEY into the environment
# After this line, Python can access it via os.environ["GROQ_API_KEY"]
# LangChain's ChatGroq automatically looks for this variable!
load_dotenv()

# 💡 TIP: If you get "API Key Not Found" error, check:
#   1. Is your .env file in the SAME folder as this script?
#   2. Did you replace "your_groq_api_key_here" with your actual key?
#   3. Get a free key at: https://console.groq.com/keys


# ============================================================
# STEP 3: Create the LLM instance
# ============================================================

# We're using Llama 3.3 70B — a powerful open-source model
# "temperature=0" means the AI gives consistent, factual answers
#   - temperature=0  → deterministic (same question = same answer)
#   - temperature=1  → creative/random (good for stories, bad for facts)
llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0
)


# ============================================================
# STEP 4: Send our first message to the AI!
# ============================================================

# We create a "HumanMessage" (just like typing in ChatGPT)
# and pass it to our LLM using the `.invoke()` method
ques = input("Enter your question: ")

print("🤖 Sending your first message to the AI...\n")


response = llm.invoke(
    [HumanMessage(content=ques)]
)


# ============================================================
# STEP 5: Print the response
# ============================================================

# The response is an AIMessage object
# `.content` gives us just the text part
print("📨 AI Response:")
print(response.content)

print("\n✅ Success! You just made your first LLM API call!")
print("🎓 Next: Module 2 — Giving the AI a personality with Prompt Templates")
