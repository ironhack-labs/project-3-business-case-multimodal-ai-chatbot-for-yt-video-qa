#!/usr/bin/env python
# coding: utf-8

"""
project_yt_chatbot_kk_final
This script builds a YouTube Q&A chatbot using Gradio, LangChain, Pinecone, and OpenAI's API.
Input dataset is the Earnings info for 8 quarters for a company (Accenture Inc., in this project).
new quarterly earnings report can be added to the input file to get updated information.
"""

# Standard library imports
import os
import getpass
import json
import time

# Third-party library imports
from torch import cuda
from dotenv import load_dotenv, find_dotenv
import yt_dlp
from whisper import load_model
from tqdm.auto import tqdm
import gradio as gr
from serpapi import GoogleSearch

# Pinecone SDK imports
from pinecone import Pinecone, ServerlessSpec

# LangChain imports
from langchain.agents import initialize_agent, Tool
from langchain.schema import HumanMessage, SystemMessage
from langchain.memory import ConversationBufferMemory
from langchain_community.chat_models import ChatOpenAI
from langchain.vectorstores import Pinecone as LangChainPinecone
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# Load Environment Variables
load_dotenv(find_dotenv())

# Securely Set API Keys and Other Environment Variables
os.environ["OPENAI_API_KEY"] = getpass.getpass("Enter OpenAI API Key: ")
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
os.environ["LANGCHAIN_PROJECT"] = "prj-yt-chatbot-qa"
os.environ["LANGCHAIN_API_KEY"] = getpass.getpass("Enter LANGCHAIN_API_KEY: ")
os.environ["PINECONE_API_KEY"] = getpass.getpass("Enter Pinecone API key: ")
os.environ["SERPAPI_API_KEY"] = getpass.getpass("Enter SerpAPI key: ")

# Device Configuration
device = f"cuda:{cuda.current_device()}" if cuda.is_available() else "cpu"

# Constants
VIDEO_LINKS_FILE = "video_links.txt"
AUDIO_OUTPUT_TEMPLATE = "audio_{}"
WHISPER_MODEL = "base"
TRANSCRIPT_DIR = "transcriptions"

# Ensure directory exists for transcriptions
os.makedirs(TRANSCRIPT_DIR, exist_ok=True)


# Functions for Loading, Downloading, Transcribing, and Vectorizing Videos
def load_video_links(filename=VIDEO_LINKS_FILE):
    """load video links from file"""
    with open(filename, "r") as file:
        return [link.strip() for link in file.readlines() if link.strip()]


# Download audio from the YT video
def download_audio(url, output_name):
    """download audio from YT videos"""
    ydl_opts = {
        "format": "bestaudio/best",
        "postprocessors": [
            {
                "key": "FFmpegExtractAudio",
                "preferredcodec": "mp3",
                "preferredquality": "192",
            }
        ],
        "outtmpl": f"{output_name}",
        "ffmpeg_location": "/usr/bin/ffmpeg",
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])
    return f"{output_name}.mp3"


# Transcribe the audio file
def transcribe_audio_with_whisper(audio_path, model_name=WHISPER_MODEL):
    """transcribe audio"""
    model = load_model(model_name)
    result = model.transcribe(audio_path)
    return result["text"]


# Transcribe all the videos
def transcribe_all_videos():
    """transcribe all videos function"""
    video_links = load_video_links(VIDEO_LINKS_FILE)
    for i, video_url in enumerate(video_links):
        print(f"Processing video {i+1}/{len(video_links)}: {video_url}")
        audio_path = download_audio(video_url, AUDIO_OUTPUT_TEMPLATE.format(i))
        transcription = transcribe_audio_with_whisper(audio_path)
        transcript_file = os.path.join(TRANSCRIPT_DIR, f"transcription_{i+1}.txt")
        with open(transcript_file, "w") as f:
            f.write(transcription)
        print(f"Completed transcription for video {i+1}")


# Pinecone Setup and Vectorization
pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
spec = ServerlessSpec(cloud="aws", region="us-east-1")
index_name = "transcription-accnt-earn-index-no-consol"
if index_name not in [index_info["name"] for index_info in pc.list_indexes()]:
    pc.create_index(name=index_name, dimension=1536, metric="dotproduct", spec=spec)
    while not pc.describe_index(index_name).status["ready"]:
        print("Waiting for index to be ready...")
        time.sleep(1)
index = pc.Index(index_name)

embedding_model = OpenAIEmbeddings(model="text-embedding-ada-002")


# Vectorization
def vectorize_individual_transcriptions(transcription_dir=TRANSCRIPT_DIR):
    """vectorize all segments"""
    all_vectorized_segments = []
    for filename in sorted(os.listdir(transcription_dir)):
        file_path = os.path.join(transcription_dir, filename)
        if not os.path.isfile(file_path) or filename.startswith("."):
            continue
        with open(file_path, "r") as f:
            transcription_text = f.read()
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1500, chunk_overlap=200
        )
        chunks = text_splitter.split_text(transcription_text)
        for chunk_idx, chunk in enumerate(chunks):
            embedding = embedding_model.embed_documents([chunk])[0]
            metadata = {
                "file_name": filename,
                "file_id": filename.replace("transcription_", "").replace(".txt", ""),
                "chunk_idx": chunk_idx,
            }
            all_vectorized_segments.append(
                {"embedding": embedding, "text": chunk, "metadata": metadata}
            )
        print(f"Vectorization complete for {filename}.")
    return all_vectorized_segments


# Upsert embeddings
def upsert_embeddings_to_pinecone(vectorized_segments, index):
    """upsert embeddings"""
    for segment in tqdm(vectorized_segments, desc="Upserting segments to Pinecone"):
        segment_id = (
            segment["metadata"]["file_id"] + "_" + str(segment["metadata"]["chunk_idx"])
        )
        embedding = segment["embedding"]
        metadata = segment["metadata"]
        metadata["text"] = segment["text"]
        try:
            index.upsert(vectors=[(segment_id, embedding, metadata)])
        except Exception as e:
            print(f"Error upserting segment {segment_id}: {e}")
    print("All segments upserted into Pinecone.")


# Execute transcription task
if __name__ == "__main__":
    transcribe_all_videos()
    # Execute the vectorization task
    vectorized_segments = vectorize_individual_transcriptions()
    # Execute the upsert task
    upsert_embeddings_to_pinecone(vectorized_segments, index)

# DEFINE ROUTER, RETRIEVE from VectorstoreDB and Retrieve from Google search functions!

# Initialize the embedding model and vector store
embedding_model = OpenAIEmbeddings(
    model="text-embedding-ada-002", openai_api_key=os.environ["OPENAI_API_KEY"]
)
chat_model = ChatOpenAI(
    model="gpt-3.5-turbo", openai_api_key=os.environ["OPENAI_API_KEY"]
)

# Pinecone index is already set up and embeddings are upserted
vector_store = LangChainPinecone(index=index, embedding=embedding_model, text_key="text")

# Initialize the router instructions
router_instructions = (
    "You are an expert at routing a user question to a vectorstore or web search.\n\n"
    "The vectorstore contains documents related to Accenture company Quarterly results, "
    "Revenue, Earnings for the last four quarters, sales, digital projects, cloud projects "
    "and employee details.\n\n"
    "Use the vectorstore for questions on these topics. For all else, and especially for "
    "current events, use web-search.\n\n"
    "Return JSON with a single key, 'datasource', that is either 'websearch' or 'vectorstore' "
    "depending on the question."
)


# Function to route query based on instructions
def route_query(query):
    """route query to capture vectordb or websearch"""
    routing_response = chat_model.invoke(
        [SystemMessage(content=router_instructions), HumanMessage(content=query)]
    )
    route_decision = json.loads(routing_response.content)
    return route_decision["datasource"]


# Function to retrieve from vector database
def retrieve_from_vectorstore(query):
    """retrieve from vectorstore"""
    retriever = vector_store.as_retriever()
    results = retriever.get_relevant_documents(query)
    document_text = "\n\n".join([doc.page_content for doc in results])

    # Summarize the retrieved documents using LLM
    prompt = (
        "Based on the following documents, provide a concise answer to the question:\n\n"
        f"Question: {query}\n\n"
        "Documents:\n"
        f"{document_text}\n\n"
        "Answer:"
    )
    answer_response = chat_model.invoke([HumanMessage(content=prompt)])
    return answer_response.content


# Function to retrieve from Google search
def web_search(query):
    """Load googlesearch parameters"""
    serp_api_key = os.environ["SERPAPI_API_KEY"]
    search = GoogleSearch(
        {
            "q": query,
            "api_key": serp_api_key,
            "num": 3,  # Limit to top 3 results for conciseness
        }
    )
    results = search.get_dict()

    # Extract relevant text from search results
    combined_text = "\n".join(
        result.get("snippet", "No snippet available")
        for result in results.get("organic_results", [])
    )

    # Summarize results using the LLM
    prompt = (
        f"Summarize the following search results to answer the question:\n\n"
        f"Question: {query}\n\n"
        f"Results:\n{combined_text}\n\n"
        "Answer:"
    )
    summary_response = chat_model.invoke([HumanMessage(content=prompt)])
    return summary_response.content


# Main function to handle the query routing and retrieval
def handle_query(query):
    """route query based on datasource"""
    datasource = route_query(query)

    if datasource == "vectorstore":
        print("Routing to Vectorstore...")
        return retrieve_from_vectorstore(query)
    elif datasource == "websearch":
        print("Routing to Web Search...")
        return web_search(query)
    else:
        return "Unknown data source. Please check the router configuration."


# DEFINE ALL THE TOOLS for ReACT agent!

# Initialize memory
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# Define the Web search tool using SerpAPI
web_search_tool = Tool(
    name="Google Search",
    func=web_search,
    description="Retrieve relevant information using google web search for general "
    "or current event questions.",
)

# Define tool for vector store retrieval
vector_store_tool = Tool(
    name="VectorStore Retrieval",
    func=retrieve_from_vectorstore,
    description="Retrieve relevant documents from the vector store for questions about "
    "Accenture's earnings, revenue, sales, projects, and employee details.",
)

# Revised prompt for the agent with improved answer formatting
agent_prompt = """
You are an expert assistant specializing in Accenture's financial data and competitor analysis.

For each user query:
1. Answer strictly based on provided context or vector data.
2. Be concise and use up to three sentences.
3. If the answer is found, respond directly by starting with "Final Answer:" without additional steps or thoughts.

Now, respond to the question:
{user_query}
"""

# Define a prompt template with placeholder for messages
prompt_template = ChatPromptTemplate.from_messages(
    [
        SystemMessage(
            content="You are a helpful assistant specializing in Accenture's financial data. Answer questions accurately."
        ),
        MessagesPlaceholder(variable_name="messages"),
    ]
)

chat_history = []

def add_to_chat_history(human_message, ai_message=None):
    """Add user and AI responses to the chat history."""
    chat_history.append(HumanMessage(content=human_message))
    if ai_message:
        chat_history.append(AIMessage(content=ai_message))


# Define the ReAct agent with the custom prompt in each tool
react_agent = initialize_agent(
    tools=[vector_store_tool, web_search_tool],
    agent="zero-shot-react-description",
    llm=chat_model,
    memory=memory,  # Integrate conversation memory
    verbose=True,
    max_iterations=5,  # Restrict maximum depth to avoid infinite loops
    handle_parsing_errors=True  # Retry on output parsing errors
)


# Main function to handle the query using the ReAct agent with the custom prompt
def handle_query_with_agent(query):
    # Format the query with the agent's prompt
    formatted_prompt = agent_prompt.format(user_query=query)
    
    # Add the user query to the chat history
    add_to_chat_history(query)

    # Use the prompt in the agent invocation
    try:
        response = react_agent({"input": formatted_prompt})
        # If a valid answer is provided, return it
        if "output" in response:
            ai_message = response["output"]
            add_to_chat_history(query, ai_message)
            return ai_message
            #return response["output"]
        else:
            # Fallback message if no answer found within the max iterations
            return "I don't know. Unable to find the answer within the provided context."
    except Exception as e:
        # In case of unexpected errors, provide a helpful message
        return f"Error encountered: {str(e)}"


# INTERACTIVE CHAT FUNCTION using ReACT Agent - REFINED 3! Works perfect! TUESDAY

# Initialize chat model and memory
chat_model = ChatOpenAI(
    model="gpt-3.5-turbo", openai_api_key=os.environ["OPENAI_API_KEY"]
)
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)


# Function to clear memory
def reset_memory():
    """Clear memory"""
    memory.clear()
    print("Memory has been reset.")


# Interactive chat function with datasource-based source tagging
def chatbot_interaction():
    """Define chatbot interaction"""
    print("Welcome to the Accenture Q&A Chatbot! Ask any question about Accenture.")
    while True:
        user_query = input("You: ")
        if user_query.lower() in ["exit", "quit", "no"]:
            print("Agent: Thank you for using the Q&A chatbot. Goodbye!")
            break

        # Determine the datasource
        datasource = route_query(user_query)

        # Tag source based on datasource
        source_label = (
            "VectorDB"
            if datasource == "vectorstore"
            else "Google Search" if datasource == "websearch" else "Unknown Source"
        )

        try:
            # Get response from the handle_query_with_agent fn to access react_agent
            answer = handle_query_with_agent(user_query)
            print(
                f"\nAgent (using {source_label}):\nQuestion: {user_query}\nAnswer: {answer}"
            )

        except Exception as e:
            print(f"Error encountered: {e}")
            print("Agent: Unable to process the request. Please try again.")

        print("\nAgent: Do you have more questions?")


# Start the chatbot interaction
chatbot_interaction()


# GRADIO only with TEXT - REFINED! Works Perfect!
def gradio_chatbot(user_query):
    """Determine the source using the routing function"""
    datasource = route_query(user_query)

    # Tag source based on the datasource returned
    source_label = (
        "VectorDB"
        if datasource == "vectorstore"
        else "Google Search" if datasource == "websearch" else "Unknown Source"
    )

    try:
        # Get response from the ReAct agent
        response = react_agent({"input": user_query})

        # Check if the response is not empty and parse the response JSON if possible
        if response and "output" in response:
            output_content = response["output"]

            # Attempt to parse JSON response if available
            try:
                parsed_response = (
                    json.loads(output_content)
                    if isinstance(output_content, str)
                    else output_content
                )
                answer = (
                    parsed_response.get("answer", output_content)
                    if isinstance(parsed_response, dict)
                    else output_content
                )
            except json.JSONDecodeError:
                answer = output_content  # Fallback to raw output if JSON parsing fails

            return f"Answer (from {source_label}):\n{answer}"

        else:
            return (
                "Error: Received an empty or unexpected response format from the agent."
            )

    except Exception as e:
        # Capture any other errors
        return f"Error encountered: {str(e)}"


# Create the Gradio Interface
iface = gr.Interface(
    fn=gradio_chatbot,  # Function to call
    inputs="text",  # Input type: text
    outputs="text",  # Output type: text
    title="Accenture Q&A Chatbot",
    description="Ask any question about Accenture and get answers with sourced information.",
)

# Launch the Gradio app
iface.launch(share=True)
