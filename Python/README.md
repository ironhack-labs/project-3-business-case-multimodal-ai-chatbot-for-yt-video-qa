# Travel ChatBot Japan - README

## Project Overview
The **Travel ChatBot Japan** is a personal travel assistant designed to help users plan their trips to Japan efficiently. It provides customized itineraries and travel recommendations based on a curated dataset of YouTube travel video transcripts. By leveraging OpenAI's language model and a structured retrieval system, the chatbot ensures reliable and high-quality travel insights.

## Features
- **Custom Itinerary Generation**: Tailored travel plans based on user preferences.
- **Personalized Recommendations**: Get insights on attractions, restaurants, and experiences.
- **Trustworthy Data Source**: Uses 140+ YouTube travel video transcripts as its knowledge base.
- **Memory Buffering**: Retains conversation context for a smoother user experience.
- **No External Source Dependency**: Responses are generated solely from the internal dataset.

## Project Architecture
1. **Data Retrieval**
   - The database consists of YouTube video transcripts from a Japan Travel playlist.
   - Utilizes `youtube_transcript_api` to retrieve and process transcripts.
   - Text is stored in a DataFrame and split into 400+ chunks for efficient retrieval.
   
2. **Chatbot Agent**
   - Implemented using `OPENAI_FUNCTIONS` for structured responses.
   - Uses `ConversationBufferMemory` (k=7) to maintain context.
   - Custom prompt ensuring the agent strictly relies on the dataset.
   
3. **Tools & Functionality**
   - Two main chatbot functionalities:
     - `GenerateItinerary`: Creates travel plans based on user input.
     - `ProvideRecommendations`: Offers travel tips and advice.
   - No integration of external web search tools to ensure dataset fidelity.

## Installation & Setup
### Prerequisites
Ensure you have the following installed:
- Python 3.8+
- `pip` package manager
- `virtualenv` (optional but recommended)

### Install Dependencies
Create and activate a virtual environment (optional but recommended):
```bash
python -m venv travelbot-env
source travelbot-env/bin/activate   # On Windows use: travelbot-env\Scripts\activate
```

Install required dependencies:
```bash
pip install -r requirements.txt
```

### Running the Chatbot
To launch the chatbot using Gradio:
```bash
python main-agent-gradio.py
```

## Usage
1. Run the chatbot script as shown above.
2. Interact with the chatbot to:
   - Get itinerary suggestions for your trip to Japan.
   - Receive personalized travel recommendations.
3. The chatbot will generate responses based on its dataset.

## Evaluation & Improvements
- The chatbot has been tested to handle non-travel-related queries by redirecting back to travel-based answers.
- Switched from `chat-conversational-react-description` to `OPENAI_FUNCTIONS` for improved stability.
- The agent sometimes supplements responses with external data to enhance itinerary plausibility.

## Future Enhancements
- **Expand Dataset**: Integrate additional trusted travel sources.
- **Improve UI/UX**: Enhance the Gradio interface for a better user experience.
- **Multilingual Support**: Enable responses in multiple languages.