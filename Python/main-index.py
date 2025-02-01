import os
from dotenv import load_dotenv, find_dotenv #
import json
import pandas as pd
from tqdm.auto import tqdm
from langchain.embeddings.openai import OpenAIEmbeddings
from pinecone import Pinecone, ServerlessSpec
from langchain.vectorstores import Pinecone

# Set up API keys
_ = load_dotenv(find_dotenv()) #
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY") or input("Enter your OpenAI API key: ")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY") or input("Enter your Pinecone API key: ")

#Load and flatten the playlist_transcripts.json file.
# Load the JSON file
with open("playlist_transcripts.json", "r") as f:
    data = json.load(f)

# Flatten the data to extract only the text
texts = []
for video_id, transcripts in data.items():
    for entry in transcripts:
        if isinstance(entry, dict):  # Ensure entry is a dictionary
            texts.append(entry["text"])  # Collect only the transcript text

# Convert to DataFrame for processing
df = pd.DataFrame({"text": texts})

# Check the structure of the DataFrame
print(df.head())

# Initialize variables for chunking
chunk_size = 4000  # Number of characters per chunk
chunk_overlap = 800  # Overlapping characters between chunks
chunks = []  # To store the resulting chunks
current_chunk = ""  # Accumulator for the current chunk

# Iterate through all rows in the DataFrame
for row in df["text"]:
    if len(current_chunk) + len(row) + 1 <= chunk_size:  # Add row if it fits in the current chunk
        current_chunk += row + " "  # Add a space between sentences
    else:
        chunks.append(current_chunk.strip())  # Save the full chunk
        current_chunk = row + " "  # Start a new chunk with overlap
        # Add overlap from the end of the previous chunk
        if chunk_overlap > 0 and len(chunks[-1]) > chunk_overlap:
            current_chunk = chunks[-1][-chunk_overlap:] + current_chunk

# Add the last chunk if it has content
if current_chunk:
    chunks.append(current_chunk.strip())

# Convert chunks into a DataFrame for embedding
chunk_df = pd.DataFrame({"text": chunks})

# Inspect the chunked DataFrame
print(f"Number of chunks: {len(chunk_df)}")
print(chunk_df.head())

# Initialize the OpenAI embedding model
embed = OpenAIEmbeddings(
    model="text-embedding-ada-002",
    openai_api_key=OPENAI_API_KEY,
)

#Connect to Pinecone and create the index
# Initialize Pinecone
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY") or input("Enter your Pinecone API key: ")
pc = Pinecone(api_key=PINECONE_API_KEY)

# Define the Pinecone index
index_name = "japan-travel-guide"  # Corrected index name to use lower case alphanumeric characters and hyphens

# Delete an existing index if the limit is reached
if len(pc.list_indexes()) >= 5:
    pc.delete_index(pc.list_indexes()[0].name)  # Extract the name from the IndexModel object

if index_name not in pc.list_indexes():
    pc.create_index(
        index_name,  # Corrected positional argument
        dimension=1536,  # Dimension for "text-embedding-ada-002"
        metric="dotproduct",
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
    )

# Connect to the index
index = pc.Index(index_name)

# Add source metadata during indexing
batch_size = 100  # Define batch size for efficiency

# Process the DataFrame in batches
for i in tqdm(range(0, len(chunk_df), batch_size)):
    i_end = min(len(chunk_df), i + batch_size)
    batch = chunk_df.iloc[i:i_end]

    # Extract chunks for embedding
    documents = batch["text"].tolist()

    # Create embeddings for the text
    embeds = embed.embed_documents(documents)

    # Generate unique IDs and add source metadata
    ids = [f"chunk-{i+j}" for j in range(len(batch))]
    metadata = [{"text": chunk, "source": f"chunk-{i+j}"} for j, chunk in enumerate(documents)]

    # Add embeddings and metadata to Pinecone
    index.upsert(vectors=zip(ids, embeds, metadata))

    # Define metadata field for text
text_field = "text"

# Initialize the vector store using the Pinecone index you've created ->  initialzie so I can access Database
vectorstore = Pinecone(
    index=index,  # Use the `index` object you already initialized in Step 2
    embedding=embed,  # Pass the embedding object
    text_key=text_field  # Key for metadata
)