import os
import shutil
from dotenv import load_dotenv
import streamlit as st
from youtube_transcript_api import YouTubeTranscriptApi

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document


# =========================
# PAGE CONFIG
# =========================
st.set_page_config(page_title="YouTube QA Chatbot", page_icon="🎥")
st.title("YouTube QA Chatbot")
st.write("Knowledge base loaded from 5 YouTube videos.")


# =========================
# LOAD ENV
# =========================
load_dotenv(dotenv_path=".env", override=True)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    st.error("OPENAI_API_KEY not found in .env")
    st.stop()

os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY


# =========================
# VIDEO IDS
# =========================
video_ids = [
    "zU9eaxjEgko",
    "HGEMscZE5dY",
    "YB49-yxy5aA",
    "Hvq_qfSJvvY",
    "Z8uANLdARj8"
]


# =========================
# TRANSCRIPT LOADING
# =========================
def get_transcripts(video_ids_list):

    all_transcripts = []

    for video_id in video_ids_list:

        try:

            transcript = YouTubeTranscriptApi().fetch(video_id)

            text = " ".join([snippet.text for snippet in transcript])
            text = text.replace("\n", " ").strip()

            all_transcripts.append(
                {
                    "video_id": video_id,
                    "text": text
                }
            )

        except Exception as e:

            st.warning(f"Could not load transcript for video {video_id}: {e}")

    return all_transcripts


# =========================
# BUILD DOCUMENTS
# =========================
def build_documents(transcripts):

    docs = []

    for item in transcripts:

        docs.append(
            Document(
                page_content=item["text"],
                metadata={"video_id": item["video_id"]}
            )
        )

    return docs


# =========================
# CHUNKING
# =========================
def chunk_documents(documents):

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    return splitter.split_documents(documents)


# =========================
# VECTOR DATABASE
# =========================
def create_vector_db(chunks):

    if os.path.exists("./chroma_db"):
        shutil.rmtree("./chroma_db")

    embeddings = OpenAIEmbeddings()

    db = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory="./chroma_db"
    )

    return db


# =========================
# BUILD QA SYSTEM
# =========================
def build_qa(vector_db):

    retriever = vector_db.as_retriever(search_kwargs={"k": 10})

    llm = ChatOpenAI(
        model="gpt-3.5-turbo",
        temperature=0
    )

    return retriever, llm


# =========================
# LOAD SYSTEM
# =========================
@st.cache_resource
def load_system():

    transcripts = get_transcripts(video_ids)

    if not transcripts:
        raise ValueError("No transcripts found")

    documents = build_documents(transcripts)

    chunks = chunk_documents(documents)

    db = create_vector_db(chunks)

    retriever, llm = build_qa(db)

    return retriever, llm, transcripts, chunks


try:

    with st.spinner("Loading videos and building database..."):

        retriever, llm, transcripts, chunks = load_system()

except Exception as e:

    st.error(f"Error while building the app: {e}")
    st.stop()


# =========================
# INFO
# =========================
st.success(f"Database ready: {len(transcripts)} videos | {len(chunks)} chunks")

with st.expander("Videos in database"):

    for vid in video_ids:

        st.write(f"https://www.youtube.com/watch?v={vid}")


# =========================
# CHAT
# =========================
question = st.text_input("Ask a question about the videos")

if st.button("Ask"):

    if question.strip():

        with st.spinner("Thinking..."):

            try:

                docs = retriever.get_relevant_documents(question)

                context = "\n\n".join(
                    [
                        f"Video ID: {doc.metadata.get('video_id')}\n{doc.page_content}"
                        for doc in docs
                    ]
                )

                prompt = f"""
You are a helpful assistant answering questions about YouTube videos.

Use the transcript context to answer the question.

If the answer is partially in the context, answer anyway.

Context:
{context}

Question:
{question}

Answer:
"""

                answer = llm.invoke(prompt)

                st.write(answer.content)

            except Exception as e:

                st.error(f"Error: {e}")

    else:

        st.warning("Write a question first.")