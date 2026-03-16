from dotenv import load_dotenv
import os
from pathlib import Path
import streamlit as st

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document

from openai import OpenAI

import matplotlib.pyplot as plt
import networkx as nx

# --------------------------------------------------
# LOAD ENV
# --------------------------------------------------

load_dotenv()

OPENAI_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAI_KEY:
    st.error("OpenAI API key not found in .env")
    st.stop()

# --------------------------------------------------
# PAGE
# --------------------------------------------------

st.set_page_config(
    page_title="HerStory Guide",
    page_icon="📚",
    layout="wide"
)

st.markdown(
"""
<style>
.stApp {
background-color: #fff9e6;
}
</style>
""",
unsafe_allow_html=True
)

# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------

with st.sidebar:

    st.title("👋 Welcome")

    st.write(
    """
    This AI assistant answers questions using
    transcripts from several YouTube videos
    about **important women in history and sport**.
    """
    )

    st.markdown("### Example questions")

    st.write("""
    • Who is Serena Williams  
    • Which women changed science  
    • Tell me about Marie Curie  
    • Which female athletes are mentioned  
    """)

# --------------------------------------------------
# TITLE
# --------------------------------------------------

st.title("📚 HerStory Guide")

st.markdown(
"Ask questions about **historical women, female athletes and influential female figures**."
)

# --------------------------------------------------
# TRANSCRIPT FOLDER
# --------------------------------------------------

TRANSCRIPTS_DIR = Path("transcripts")

# --------------------------------------------------
# LOAD TRANSCRIPTS
# --------------------------------------------------

def load_transcripts():

    transcripts = []

    files = list(TRANSCRIPTS_DIR.glob("*.txt"))

    for file in files:

        text = file.read_text(encoding="utf-8")

        transcripts.append(
            Document(
                page_content=text,
                metadata={"video": file.stem}
            )
        )

    return transcripts

# --------------------------------------------------
# VECTOR DATABASE
# --------------------------------------------------

@st.cache_resource
def create_vector_db():

    docs = load_transcripts()

    if len(docs) == 0:
        st.error("No transcripts found in /transcripts folder")
        st.stop()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    chunks = splitter.split_documents(docs)

    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_KEY)

    db = Chroma.from_documents(
        chunks,
        embedding=embeddings,
        persist_directory="chroma_db"
    )

    return db

# --------------------------------------------------
# LOAD DB
# --------------------------------------------------

vector_db = create_vector_db()

retriever = vector_db.as_retriever(search_kwargs={"k":4})

llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0,
    openai_api_key=OPENAI_KEY
)

client = OpenAI(api_key=OPENAI_KEY)

# --------------------------------------------------
# DIAGRAM
# --------------------------------------------------

def create_diagram(topic):

    prompt = f"""
Create simple concept relations about {topic}

Format:

Concept -> Idea
Idea -> Impact
Impact -> Result
"""

    text = llm.invoke(prompt).content

    G = nx.DiGraph()

    for line in text.split("\n"):

        if "->" in line:

            a,b = line.split("->")

            G.add_edge(a.strip(), b.strip())

    pos = nx.spring_layout(G)

    fig = plt.figure(figsize=(7,5))

    nx.draw(
        G,
        pos,
        with_labels=True,
        node_size=2500,
        node_color="#fff2cc",
        font_size=10
    )

    return fig

# --------------------------------------------------
# QUESTION
# --------------------------------------------------

question = st.text_input("Ask a question")

# --------------------------------------------------
# RAG
# --------------------------------------------------

if question:

    docs = retriever.get_relevant_documents(question)

    context = "\n\n".join([d.page_content for d in docs])

    prompt = f"""
Answer using ONLY the transcript information.

Context:
{context}

Question:
{question}
"""

    answer = llm.invoke(prompt).content

    st.subheader("Answer")
    st.write(answer)

    # -------------------------
    # VOICE
    # -------------------------

    try:

        speech = client.audio.speech.create(
            model="gpt-4o-mini-tts",
            voice="alloy",
            input=answer
        )

        st.audio(speech.content)

    except:
        st.warning("Voice generation failed")

    # -------------------------
    # DIAGRAM
    # -------------------------

    st.subheader("Concept Diagram")

    try:
        fig = create_diagram(question)
        st.pyplot(fig)
    except:
        st.warning("Diagram generation failed")