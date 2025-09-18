# Personal Finance Buddy

An AI-powered **financial coaching assistant** that helps users build savings/house/trip plans, evaluate financial health, and answer personal finance questions with context from YouTube transcripts, retrieval-augmented generation (RAG), and live web search.

## 🚀 Features
- **Data Ingestion**:  
  - Downloads YouTube audio with `yt-dlp`.  
  - Transcribes/ translates to English with OpenAI Whisper.  
  - Cleans & chunks transcripts (~500 characters).  
  - Embeds with OpenAI embeddings and stores in **ChromaDB**.

- **RAG Setup**:  
  - Uses GPT-4o as the LLM with low temperature for consistent answers.  
  - Query expansion with **MultiQueryRetriever**.  
  - **Cohere Rerank** + Contextual Compression for precise retrieval.  
  - History-aware retriever to refine based on conversation context.  
  - Conversation memory with **RunnableWithMessageHistory**.  
  - Full LangSmith tracing and evaluation.

- **Agent Setup**:  
  - Tools:  
    - `create_plan` → builds custom house/trip/savings plans step by step.  
    - `web_search` → fetches live financial info (mortgage rates, etc.).  
  - Financial health evaluation: Strong / Moderate / Weak with explanation.  
  - Supports multi-turn conversations and profile persistence.  
  - Translation support with **deep-translator**.

- **Evaluation**:  
  - Automated tests (self-tests + LangSmith dataset).  
  - Evaluates for correctness, clarity, and helpfulness.

## 🛠️ Tech Stack
- **LangChain** (chains, retrievers, agents, memory).  
- **OpenAI (GPT-4o, embeddings)**.  
- **ChromaDB** for vector storage.  
- **Whisper** for transcription/translation.  
- **yt-dlp** for YouTube audio.  
- **deep-translator** for translation.  
- **DuckDuckGoSearch** for web search.  
- **LangSmith** for evaluation and tracing.  

## 📂 Project Structure
- **Cell 1**: Data ingestion (YouTube → Whisper → ChromaDB).  
- **Cell 2**: RAG setup (retrievers, reranking, memory).  
- **Cell 3**: Agent setup (tools, financial plans, web search).  
- **Cell 4**: Evaluation (self-tests + LangSmith dataset).  

## ▶️ How to Run
1. Clone the repo / open notebook.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Add `.env` with keys:
   ```env
   OPENAI_API_KEY=your-openai-key
   COHERE_API_KEY=your-cohere-key
   LANGCHAIN_API_KEY=your-langsmith-key
   LANGCHAIN_PROJECT=Personal-Finance-Buddy
   ```
4. Run notebook cells in order:
   - Cell 1 → build transcripts + Chroma DB  
   - Cell 2 → RAG setup  
   - Cell 3 → Agent setup  
   - Cell 4 → Evaluation  
