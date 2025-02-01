from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains import RetrievalQA
import os 
from dotenv import load_dotenv, find_dotenv
from pinecone import Pinecone
from langchain.agents import Tool 
from langchain.agents import initialize_agent
from langchain.prompts import SystemMessagePromptTemplate
import gradio as gr
from langchain_core.prompts import SystemMessagePromptTemplate
from langchain.chains.conversation.memory import ConversationBufferMemory
from langchain.agents import initialize_agent, AgentType

# Load API keys and initialize
_ = load_dotenv(find_dotenv())
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY") or input("Enter your OpenAI API key: ")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY") or input("Enter your Pinecone API key: ")

# Initialize the OpenAI Embeddings model
embedding_model = OpenAIEmbeddings(model="text-embedding-ada-002", api_key=OPENAI_API_KEY)

pc = Pinecone(api_key=PINECONE_API_KEY)
index_name = "japan-travel-guide"

index = pc.Index(index_name)

    # Create Pinecone vector store
vector_store = PineconeVectorStore(
    index=index,
    embedding=embedding_model,
    #text_key="text_excerpt"  # Update this if your metadata key is different
    )

retriever = vector_store.as_retriever()

# Configure ChatOpenAI and the Retrieval Chain
llm = ChatOpenAI(model="gpt-4o-mini", api_key=OPENAI_API_KEY)
qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever, chain_type="stuff")

#Custom Prompt Template
system_prompt = SystemMessagePromptTemplate.from_template(
    """
    You are a friendly and knowledgeable travel assistant specialized in planning trips to Japan. Your role is to create travel plans, personalized itineraries, and provide tailored recommendations for activities, attractions, and dining options in Japan.

    **Rules and Workflow:**
    1. **Trip Planning:**
       - Use tools to generate itineraries and answer questions based on the user’s preferences, such as travel duration, interests, and budget.
       - Always prioritize providing accurate and specific advice.

    2. **Recommendations:**
       - Recommend attractions, activities, and dining options tailored to the user's preferences.
       - Provide cultural insights, tips, and advice for visiting Japan.

    3. **Tool Usage:**
       - Always use tools to retrieve relevant information or generate itineraries.
       - Your responses must rely on tool outputs. Do not use your own reasoning beyond the tool's output.

    **Critical Instructions:**
    - Never respond without using tools for relevant data.
    - Avoid making assumptions or speculations.
    - Always follow user input closely to customize your recommendations.

    **Memory Context:**
    - Here is the previous conversation: `{chat_history}`

    Follow these guidelines strictly to ensure helpful, tool-based assistance for the user.
    """
)

# Tool 1: Generate Itinerary
def generate_itinerary_tool(user_input: str) -> str:
    """
    Generates a personalized itinerary based on user input.
    """
    # Use your existing logic for itinerary generation
    itinerary = qa_chain.run(user_input)
    return itinerary

# Tool 2: Provide Recommendations
def provide_recommendations_tool(user_input: str) -> str:
    """
    Provides recommendations for activities, attractions, and dining options in Japan.
    """
    # Use your existing logic for recommendations
    recommendations = qa_chain.run(user_input)
    return recommendations

tools = [
    Tool(
        name="GenerateItinerary",
        func=generate_itinerary_tool,
        description="Use this tool to create personalized itineraries for travel in Japan. Input: trip details like duration, interests, and locations."
    ),
    Tool(
        name="ProvideRecommendations",
        func=provide_recommendations_tool,
        description="Use this tool to recommend activities, attractions, and dining options in Japan. Input: specific preferences or questions."
    )
]

# Initialize the memory context for the conversation
conversational_memory = ConversationBufferMemory(
    memory_key="chat_history",
    k=7,  # Number of previous interactions to remember
    return_messages=True
)

# Initialize agent
agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.OPENAI_FUNCTIONS,
    verbose=True,
    memory=conversational_memory,
    max_iterations=3,
    agent_kwargs={
        "system_message": system_prompt,
        "memory_key": "chat_history",
    },
)

# Deploy in Gradio Interface
# Store conversation history
chat_history = []

# Function to handle user input and return chatbot's response
def chatbot_response(user_input):
    global chat_history
    try:
        # Generate response from LangChain agent
        response = agent.invoke({"input": user_input}).get("output", "I didn't understand that. Please try again.")
    except Exception as e:
        response = f"An error occurred: {str(e)}"

    # Append user input and bot response to the chat history
    chat_history.append((user_input, response))
    return chat_history  # Return updated chat history for display

# Function to reset LangChain memory
def reset_agent_memory():
    global memory, agent
    memory = ConversationBufferMemory(memory_key="chat_history", k=7, return_messages=True)
    agent = initialize_agent(
        tools=tools,
        llm=llm,
        agent=AgentType.OPENAI_FUNCTIONS,
        verbose=True,
        memory=memory,
        max_iterations=3,
        agent_kwargs={
            "system_message": system_prompt,
            "memory_key": "chat_history",
        },
    )

# Function to clear chat
def clear_chat():
    global chat_history
    chat_history = []  # Reset Gradio chat history
    reset_agent_memory()  # Reset LangChain memory
    return []  # Clear Gradio chatbot UI

# Define the Gradio interface
with gr.Blocks() as demo:
    gr.Markdown("# 🗾 Japan Travel Chatbot")
    gr.Markdown("Ask me anything about traveling in Japan, from itineraries to cultural insights!")

    # Chatbot interface
    chatbot = gr.Chatbot(label="Chat History")

    with gr.Row():
        user_input = gr.Textbox(label="Your Question", placeholder="Type your question here...", lines=1)
        submit_button = gr.Button("Submit")
    
    # Clear chat button
    clear_button = gr.Button("Clear Chat")

    # Connect the buttons to functions
    submit_button.click(chatbot_response, inputs=[user_input], outputs=[chatbot])
    clear_button.click(clear_chat, inputs=None, outputs=[chatbot])

# Launch the app
demo.launch(share=True)
