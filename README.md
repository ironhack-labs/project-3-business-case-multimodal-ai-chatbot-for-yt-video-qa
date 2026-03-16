![logo_ironhack_blue 7](https://user-images.githubusercontent.com/23629340/40541063-a07a0a8a-601a-11e8-91b5-2f13e4e6b441.png)

# YouTube Video Question Answering Chatbot (RAG System)

## Project Overview

This project implements an AI chatbot capable of answering questions about the content of a YouTube video.
The system extracts the transcript from a video, processes the text, stores semantic embeddings in a vector database, and uses a Retrieval Augmented Generation (RAG) pipeline to generate answers to user queries.

Users interact with the chatbot through a simple conversational interface built with Streamlit.

The goal of the project is to demonstrate how modern NLP tools such as LangChain, vector databases, and large language models can be combined to build a practical AI assistant for multimedia content.

---

# Business Case

Video content is widely used for education, tutorials, and customer support, but searching inside videos can be difficult.

This chatbot improves accessibility and usability of video content by allowing users to ask natural language questions and receive relevant answers extracted directly from the video transcript.

Possible applications include:

* Educational content navigation
* Customer support automation
* Knowledge extraction from long videos
* Accessibility for users who prefer reading instead of watching videos

---

# Project Architecture

The project follows a Retrieval Augmented Generation (RAG) architecture composed of several components:

1. **YouTube Transcript Extraction**
   The transcript of a YouTube video is retrieved using the `youtube-transcript-api`.

2. **Text Processing**
   The transcript text is cleaned and split into smaller chunks suitable for semantic search.

3. **Embeddings Generation**
   Each text chunk is converted into vector embeddings using a language model.

4. **Vector Database Storage**
   Embeddings are stored in a vector database (ChromaDB or FAISS).

5. **Information Retrieval**
   When a user asks a question, the system retrieves the most relevant transcript chunks.

6. **Answer Generation**
   A Large Language Model generates an answer using the retrieved context.

7. **Chat Interface**
   Users interact with the system through a Streamlit web application.

---

# Technologies Used

* Python
* LangChain
* OpenAI / HuggingFace models
* ChromaDB or FAISS (vector database)
* youtube-transcript-api
* yt-dlp
* Streamlit
* LangSmith (optional for evaluation and monitoring)

---

# How the System Works

1. The system retrieves the transcript of a YouTube video.
2. The transcript is cleaned and split into text chunks.
3. Each chunk is converted into embeddings.
4. The embeddings are stored in a vector database.
5. When a user asks a question:

   * the system retrieves the most relevant chunks from the database
   * the LLM generates a response using the retrieved information.

This process allows the chatbot to answer questions based specifically on the content of the video.

---

# Installation

Clone the repository and install the required dependencies.

```bash
git clone https://github.com/your-username/project-name.git

cd project-name

python -m venv venv

venv\Scripts\activate

pip install -r requirements.txt
```

---

# Running the Application

To start the chatbot interface:

```bash
streamlit run app.py
```

This will open a local web interface where users can interact with the chatbot.

---

# Example Usage

Examp

