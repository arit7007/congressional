# Hear Me: Your Chat Companion with Azure Text-to-Speech and OpenAI GPT Integration

**Hear Me** is a conversational AI companion powered by OpenAI for text responses and Azure Cognitive Service for natural-sounding text-to-speech synthesis. The app also includes sentiment analysis, and some content moderation.

## Features

- **AI-Powered Chatbot**: Uses OpenAI GPT to generate empathetic responses to the user's message.
- **Sentiment Analysis**: Analyzes the user's message sentiment (positive, negative, or neutral) and adjusts responses accordingly.
- **Azure Text-to-Speech**: Converts text responses into speech using Azure’s text-to-speech cognitive services.
- **Moderation Check**: Screens user messages to ensure compliance with content policies and filter inappropriate content, while maintaining empathetic response.
- **Session Management**: Maintains a conversation history with customizable session duration (default 30 minutes). In the future, better memorization will be added
- **Prompt Injection**: Check for prompt injection, and override with system default (TBD).

### Prerequisites

- **Python**: Version 3.x
- **Required Libraries**: Install via `pip install -r requirements.txt`
  - `openai`
  - `flask`
  - `requests`
  - `markupsafe`
  - `azure-cognitiveservices-speech`

### Environment Variables

Set up the following environment variables before running the app:

- `OPENAI_API_KEY`: Your OpenAI API key.
- `AZURE_SPEECH_KEY`: Your Azure Speech API key.
- `AZURE_SERVICE_REGION`: Your Azure Speech service region.

### Running the application
Clone the repository and navigate to the project folder.
Install any dependencies.
Start the application:

python cm_with_azure_tts.py

The server will run on http://127.0.0.1:5000.

Once the app is running, click the link and chat with your new friendly buddy!!
