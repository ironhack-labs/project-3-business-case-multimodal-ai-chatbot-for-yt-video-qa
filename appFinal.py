from dotenv import load_dotenv
import os
from pathlib import Path
import re
import base64

import streamlit as st

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from openai import OpenAI


# --------------------------------------------------
# ENV
# --------------------------------------------------

load_dotenv()

OPENAI_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAI_KEY:
    st.error("Missing OpenAI API key")
    st.stop()


# --------------------------------------------------
# PAGE STYLE
# --------------------------------------------------

st.set_page_config(
    page_title="HerStory Guide",
    page_icon="📚",
    layout="wide"
)

st.markdown("""
<style>

.stApp {
background-color:#fff7dc;
}

h1 {
color:#5c4b00;
}

</style>
""", unsafe_allow_html=True)


# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------

with st.sidebar:

    st.title("📚 HerStory Guide")

    st.markdown("""
Ask questions about **women in history**.

Examples:

• Who was Marie Curie  
• Which women changed science  
• Tell me about Joan of Arc  
• What did Florence Nightingale do  

The assistant answers **only using transcript data.**
""")


# --------------------------------------------------
# TITLE
# --------------------------------------------------

st.title("📚 HerStory Guide")


# --------------------------------------------------
# MEMORY
# --------------------------------------------------

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []


# --------------------------------------------------
# CONSTANTS
# --------------------------------------------------

TRANSCRIPTS_DIR = Path("transcripts")


# --------------------------------------------------
# HELPERS
# --------------------------------------------------

def clean_text(text):

    text = text.replace("\n", " ")
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def load_transcripts():

    docs = []

    files = list(TRANSCRIPTS_DIR.glob("*.txt"))

    for f in files:

        text = f.read_text(encoding="utf-8")

        docs.append(
            Document(
                page_content=clean_text(text),
                metadata={"video": f.stem}
            )
        )

    return docs


# --------------------------------------------------
# VECTOR DB
# --------------------------------------------------

@st.cache_resource
def build_vector_db():

    docs = load_transcripts()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    chunks = splitter.split_documents(docs)

    embeddings = OpenAIEmbeddings()

    db = Chroma.from_documents(
        chunks,
        embeddings
    )

    return db, docs


vector_db, all_docs = build_vector_db()

retriever = vector_db.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 8, "fetch_k": 20}
)


# --------------------------------------------------
# MODELS
# --------------------------------------------------

llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0.2
)

client = OpenAI()


# --------------------------------------------------
# ENTITY EXTRACTION
# --------------------------------------------------

def extract_entity(question):

    prompt = f"""
Extract the name of a person mentioned in this question.

Question:
{question}

Return only the name.
If none exists return NONE.
"""

    r = llm.invoke(prompt).content.strip()

    if r == "NONE":
        return None

    return r


# --------------------------------------------------
# RETRIEVAL
# --------------------------------------------------

def retrieve_docs(question):

    entity = extract_entity(question)

    if entity:

        entity_lower = entity.lower()

        matches = [
            d for d in all_docs
            if entity_lower in d.page_content.lower()
        ]

        if matches:
            return matches[:5]

    return retriever.invoke(question)


# --------------------------------------------------
# CONTEXT
# --------------------------------------------------

def build_context(docs):

    blocks = []

    for i, d in enumerate(docs):

        blocks.append(
f"""
SOURCE {i+1}

{d.page_content}
"""
)

    return "\n".join(blocks)


# --------------------------------------------------
# ANSWER
# --------------------------------------------------

def answer_question(question):

    docs = retrieve_docs(question)

    context = build_context(docs)

    history = "\n".join(st.session_state.chat_history)

    prompt = f"""
You are a research assistant.

Conversation history:
{history}

Use the transcript excerpts to answer the question.

If the answer is implied you may infer it.

Context:
{context}

Question:
{question}
"""

    response = llm.invoke(prompt).content

    st.session_state.chat_history.append(f"User: {question}")
    st.session_state.chat_history.append(f"Assistant: {response}")

    return response


# --------------------------------------------------
# SIMPLE IMAGE
# --------------------------------------------------

def generate_image(question):

    try:

        img = client.images.generate(
            model="gpt-image-1",
            prompt=f"simple educational illustration of {question}",
            size="1024x1024"
        )

        image_base64 = img.data[0].b64_json
        image_bytes = base64.b64decode(image_base64)

        return image_bytes

    except:

        return None


# --------------------------------------------------
# AUDIO
# --------------------------------------------------

def generate_audio(text):

    try:

        speech = client.audio.speech.create(
            model="gpt-4o-mini-tts",
            voice="alloy",
            input=text
        )

        return speech.content

    except Exception as e:

        print(e)
        return None


# --------------------------------------------------
# UI
# --------------------------------------------------

try:

    question = st.text_input("Ask a question about women in history")

    if question:

        answer = answer_question(question)

        st.subheader("Answer")
        st.write(answer)

        if st.button("🔊 Read Answer"):

            audio = generate_audio(answer)

            if audio:
                st.audio(audio)
            else:
                st.warning("Audio generation failed")

except Exception as e:

    st.error("An error occurred")
    st.write(e)