# 📄 PDF Q&A Chatbot — GenAI RAG Project

A step-by-step **Retrieval Augmented Generation (RAG)** chatbot that lets you upload any PDF and ask questions about it. Built with **Groq + LangChain + ChromaDB + Streamlit**.

## 🏗️ Architecture

```
User Question → Embedding → ChromaDB (similarity search) → Relevant Chunks → LLM → Answer
```

## 📂 Project Structure (Module-wise)

| Module | File | What It Teaches |
|--------|------|-----------------|
| Setup | `requirements.txt`, `.env`, `.gitignore` | Dependencies & secrets management |
| 1 | `module_01_llm_connection.py` | First LLM API call with Groq + python-dotenv |
| 2 | `module_02_prompt_template.py` | Prompt engineering & system messages |
| 3 | `module_03_pdf_loader.py` | PDF loading & text chunking with overlap |
| 4 | `module_04_vector_store.py` | Embeddings + persistent ChromaDB |
| 5 | `module_05_rag_chain.py` | Full RAG pipeline + hallucination guard |
| 6 | `app.py` | Streamlit UI with session state |

## 🚀 Quick Setup

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Add your Groq API key to .env
#    Get free key at: https://console.groq.com/keys

# 3. Run modules one by one (learning path)
python module_01_llm_connection.py
python module_02_prompt_template.py
python module_03_pdf_loader.py    # needs a sample.pdf
python module_04_vector_store.py  # needs a sample.pdf
python module_05_rag_chain.py     # needs a sample.pdf

# 4. Run the final app
streamlit run app.py
```

## 🔑 Get Your Free API Key

1. Go to [console.groq.com/keys](https://console.groq.com/keys)
2. Sign up / log in
3. Click **"Create API Key"**
4. Copy the key and paste it in your `.env` file

## 🧠 Key Concepts Covered

- **LLM (Large Language Model)** — AI models that understand and generate text
- **Prompt Engineering** — Controlling AI behavior with system messages
- **Token Limits** — Why we can't send entire documents at once
- **Text Chunking** — Breaking documents into manageable pieces
- **Chunk Overlap** — Preserving context across chunk boundaries
- **Embeddings** — Converting text to numerical vectors
- **Vector Database** — Storing and searching by meaning (not keywords)
- **RAG** — Combining retrieval with generation for grounded answers
- **Hallucination Prevention** — Stopping the AI from making up facts
- **Session State** — Preserving data across Streamlit reruns

## ☁️ Deployment (Streamlit Cloud)

1. Push your code to GitHub (make sure `.env` is in `.gitignore`!)
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub repo
4. Add `GROQ_API_KEY` in the **Secrets** section
5. Deploy!

## 🤝 WeCodark Community

Fork this repo and build your own version:
- 📚 **Legal AI** — Upload legal contracts, ask questions
- 🏥 **Medical AI** — Upload research papers, get summaries
- 💰 **Finance AI** — Upload annual reports, extract insights
