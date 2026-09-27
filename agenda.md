Phase 1: The Context & The "Why"
01. Intro & Demo: Show the final product. Drag a 20-page PDF into the UI and ask a specific question.

02. AI vs. GenAI: Simple explanation. AI predicts (Is this a cat?); GenAI creates (Draw a cat in a space suit).

03. Why a "Product" Project? Explain that companies today don’t want "toy" scripts; they want end-to-end applications with a UI.

Phase 2: The Core "Brain" (Step-by-Step Code)
04. Step 1: Connecting to the LLM (Groq + LangChain)

The Code: Writing the first ChatGroq invocation.

⚠️ The Problem: "API Key Not Found" or accidentally hardcoding the key and pushing it to GitHub.

✅ The Solution: Introduce python-dotenv. Show how to create a .env file to keep secrets safe.

05. Step 2: Giving the AI a Personality (Prompt Templates)

The Code: Creating a PromptTemplate where the AI is told it is a "Senior Research Analyst."

⚠️ The Problem: The AI gives "Introductory fluff" (e.g., "Sure, I can help with that! Here is the answer...") when we only want the data.

✅ The Solution: Using "System Messages" and strict instructions like "Return ONLY the answer without a preamble."

Phase 3: The Architecture & Data (RAG)
06. Architecture Reveal: Use a diagram to show how data flows from a PDF into the Vector DB and then to the LLM.

07. Step 3: PDF Loading & The "Chunking" Problem

The Code: Using PyPDFLoader to read the file.

⚠️ The Problem: Loading a 100-page book all at once. The LLM hits its "Token Limit" and crashes.

✅ The Solution: Text Splitting. Use RecursiveCharacterTextSplitter. Explain "Overlap"—why we need the end of one chunk to match the start of the next so we don't lose context.

08. Step 4: The Vector Store (ChromaDB)

The Code: Converting chunks into "Embeddings" and saving them to ChromaDB.

⚠️ The Problem: Every time you restart the script, the database is deleted (In-memory issue).

✅ The Solution: Use PersistentClient in ChromaDB so the data stays on your hard drive forever.

Phase 4: Making it a "Product" (The UI)
09. Step 5: Building the UI with Streamlit

The Code: Creating the st.file_uploader and st.chat_input.

⚠️ The Problem: "The Refresh Loop." In Streamlit, every time you click a button, the whole code runs again from the top, losing your chat history.

✅ The Solution: Session State. Show how to use st.session_state to "save" the chat history so it feels like a real conversation.

10. Step 6: Handling "Hallucinations"

⚠️ The Problem: You ask a question NOT in the PDF, and the AI starts lying (making up facts).

✅ The Solution: Update the prompt: "If the answer is not in the provided context, strictly say 'I don't know'."

Phase 5: Conclusion & "WeCodark" Community
11. Deployment: Briefly mention Streamlit Cloud for hosting.

12. Closing: Ask viewers to fork the "WeCodark" repo and build their own version (Legal AI, Medical AI, etc.).