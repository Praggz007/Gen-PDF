"""
============================================================
🧠 MODULE 2: Giving the AI a Personality (Prompt Templates)
============================================================

📖 WHAT WE'RE DOING:
    In Module 1, we just asked a raw question. The AI responded however
    it wanted — sometimes with fluff like "Sure! I'd be happy to help..."
    
    In this module, we CONTROL how the AI behaves by giving it:
      1. A ROLE (System Message) — "You are a Senior Research Analyst"
      2. STRICT RULES — "Return ONLY the answer, no preamble"
      3. A TEMPLATE — A reusable format for all our questions

📖 WHAT IS A PROMPT TEMPLATE?
    Imagine you're writing an email. Instead of typing the whole thing
    every time, you use a template:
        "Dear {name}, Thank you for your order of {product}..."

    Prompt Templates work the same way! We create a template with
    placeholders like {topic}, and LangChain fills them in for us.

📖 SYSTEM vs HUMAN MESSAGES:
    - SystemMessage: Instructions to the AI about HOW to behave
      (The user never sees this — it's like whispering to the AI)
    - HumanMessage: The actual question from the user

⚠️ THE PROBLEM:
    Without a system message, the AI gives "fluffy" responses:
      "Sure, I can help with that! Here is a great explanation..."
    We want JUST the data, no fluff.

✅ THE SOLUTION:
    System message: "Return ONLY the answer without any preamble,
    introduction, or conversational filler."

▶️ HOW TO RUN:
    python module_02_prompt_template.py
"""

# ============================================================
# STEP 1: Imports
# ============================================================

from dotenv import load_dotenv
from langchain_groq import ChatGroq

# ChatPromptTemplate lets us create reusable prompt templates
# with System + Human message pairs
from langchain_core.prompts import ChatPromptTemplate

# StrOutputParser converts the AI's response object into a plain string
# Without this, we'd get an AIMessage object instead of just text
from langchain_core.output_parsers import StrOutputParser

# Load the API key from .env
load_dotenv()


# ============================================================
# STEP 2: Create the LLM (same as Module 1)
# ============================================================

llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0
)


# ============================================================
# STEP 3: Create a Prompt Template
# ============================================================

# This is the KEY concept of this module!
# We define TWO messages:
#   1. "system" — tells the AI WHO it is and HOW to respond
#   2. "human"  — the actual question (with a {topic} placeholder)

prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """You are a Senior Research Analyst with 20 years of experience.
Your job is to provide clear, concise, and accurate information.

STRICT RULES:
- Return ONLY the answer without any preamble or introduction
- Do NOT start with "Sure", "Of course", "Great question" etc.
- Be direct and factual
- Use bullet points when listing multiple items
- Keep the response under 100 words"""
    ),
    (
        "human",
        "Explain the following topic: {topic}"
    )
])

# 💡 WHY THIS MATTERS:
#   Without the system message, if you ask "What is RAG?", you might get:
#     "Sure! I'd love to explain RAG. RAG stands for..."
#
#   WITH the system message, you get:
#     "RAG (Retrieval Augmented Generation) is a technique that..."
#   
#   Much cleaner! This is called "Prompt Engineering."


# ============================================================
# STEP 4: Create a Chain (Prompt → LLM → Output Parser)
# ============================================================

# This is LangChain's "pipe" operator — it connects components:
#   prompt → fills in the template
#   llm    → sends it to the AI
#   StrOutputParser() → extracts just the text from the response
#
# Think of it like a factory assembly line:
#   Raw materials → Machine 1 → Machine 2 → Machine 3 → Final product

chain = prompt | llm | StrOutputParser()


# ============================================================
# STEP 5: Run the chain!
# ============================================================

# We pass in our variable {topic} and the chain does everything:
#   1. Fills "Retrieval Augmented Generation" into the {topic} slot
#   2. Sends the formatted prompt to Groq's LLM
#   3. Parses the response into a clean string

topic = input("Enter your topic: ")

print("🤖 Asking the AI (with personality) about RAG...\n")

response = chain.invoke({"topic": topic})

print("📨 AI Response:")
print(response)


# ============================================================
# BONUS: Try with a different topic!
# ============================================================

another_topic = input("Enter another topic: ")
print("\n" + "="*50)
print("🤖 Now asking about Vector Databases...\n")

response2 = chain.invoke({"topic": another_topic})

print("📨 AI Response:")
print(response2)

print("\n✅ Notice how the AI responds directly without any fluff!")
print("🎓 Next: Module 3 — Loading PDFs and splitting them into chunks")
