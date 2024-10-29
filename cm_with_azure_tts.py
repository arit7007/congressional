import logging
import os
import random
import secrets
import time
from datetime import timedelta

import openai
import requests
from azure.cognitiveservices.speech import SpeechConfig, SpeechSynthesizer, AudioConfig
from flask import Flask, request, jsonify, render_template, session
from markupsafe import escape

app = Flask(__name__)

logging.basicConfig(level=logging.INFO)

# Generate a secure SECRET_KEY
app.config['SECRET_KEY'] = secrets.token_urlsafe(32)
app.config['PERMANENT_SESSION_LIFETIME'] = timedelta(minutes=30)  # set session timeout to 30 minutes

openai.api_key = os.getenv('OPENAI_API_KEY')

# Ensure the OpenAI API key is set
if openai.api_key is None:
    raise ValueError("Please set the OPENAI_API_KEY environment variable.")

# Azure Speech Service configuration
azure_speech_key = os.getenv('AZURE_SPEECH_KEY')
azure_service_region = os.getenv('AZURE_SERVICE_REGION')

if azure_speech_key is None or azure_service_region is None:
    raise ValueError("Please set the AZURE_SPEECH_KEY and AZURE_SERVICE_REGION environment variables.")

speech_config = SpeechConfig(subscription=azure_speech_key, region=azure_service_region)
audio_config = AudioConfig(use_default_microphone=True)

# Define default language and voices
LANGUAGE_VOICES = {
    'english': 'en-US-JennyNeural',
    'spanish': 'es-ES-ElviraNeural',
    'french': 'fr-FR-DeniseNeural',
    'german': 'de-DE-ConradNeural',
}


def extract_user_name(conversation_history):
    """Extract user's name if available from chat."""
    name_prompt = [
        {"role": "system",
         "content": "You are an assistant that extracts names from user conversations. If there is no name, respond with 'None'."},
        {"role": "user", "content": f"Extract the name from this conversation: {conversation_history}"}
    ]

    try:
        response = openai.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=name_prompt
        )
        name = response.choices[0].message.content.strip()
        if name.lower() == "none":
            return None
        return name
    except Exception as e:
        logging.error(f"Error extracting user name: {e}")
        return None


def analyze_sentiment_gpt(user_message):
    """Analyze sentiment of the user's message to customize the response."""
    sentiment_prompt = [
        {"role": "system", "content": "You are a helpful assistant that analyzes the sentiment of the user's message."},
        {"role": "user",
         "content": f"Analyze the sentiment of this message: '{user_message}'. Reply with one word: Positive, Negative, or Neutral."}
    ]

    try:
        response = openai.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=sentiment_prompt
        )
        sentiment = response.choices[0].message.content.strip().lower()
        if sentiment in ["positive", "negative", "neutral"]:
            return sentiment
        else:
            return "neutral"  # Default to neutral if the response is unexpected
    except Exception as e:
        logging.error(f"Error analyzing sentiment: {e}")
        return "neutral"  # Default to neutral in case of an error


@app.route('/get-voices', methods=['GET'])
def get_voices():
    url = f'https://{azure_service_region}.tts.speech.microsoft.com/cognitiveservices/voices/list'
    headers = {
        'Ocp-Apim-Subscription-Key': azure_speech_key,
    }

    response = requests.get(url, headers=headers)

    if response.status_code == 200:
        voices = response.json()
        voices_data = [{"name": voice["Name"], "displayName": voice["DisplayName"], "lang": voice["Locale"]} for voice in voices]
        return jsonify(voices_data)
    else:
        logging.error(f"Error fetching voices: {response.status_code}, {response.text}")
        return jsonify({"error": "Unable to fetch voice list."}), 500


def generate_response(messages, model="gpt-3.5-turbo", retry_count=3):
    for attempt in range(retry_count):
        try:
            response = openai.chat.completions.create(
                model=model,
                messages=messages
            )
            return response.choices[0].message.content.strip()
        except openai.error.OpenAIError as e:
            logging.error(f"OpenAI API error on attempt {attempt + 1}: {e}")
            time.sleep(2)  # Wait before retrying
        except Exception as e:
            logging.error(f"Unexpected error on attempt {attempt + 1}: {e}")
    return "I'm sorry, but I'm having trouble processing your request right now. Please try again later."


def synthesize_speech(text, voice="en-US-JennyNeural"):
    try:
        # Set the speech synthesis voice based on the selected voice
        speech_config.speech_synthesis_voice_name = voice
        synthesizer = SpeechSynthesizer(speech_config=speech_config, audio_config=audio_config)
        synthesizer.speak_text_async(text)
    except Exception as e:
        logging.error(f"Azure Text-to-Speech error: {e}")


@app.route('/')
def home():
    return render_template('cm_with_azure_tts_index.html')


@app.route('/chat', methods=['POST'])
def chat():
    try:
        user_message = escape(request.json.get('message', '').strip())
        selected_voice = request.json.get('voice', 'en-US-JennyNeural')
        assistant_name = request.json.get('assistant_name', 'Jenny')
        preferred_language = request.json.get('language', 'english').lower()
    except Exception as e:
        logging.error(f"Error parsing user message: {e}")
        return jsonify({"response": "Invalid input."}), 400

    if not user_message:
        return jsonify({"response": "Please enter a message."}), 400

    # Analyze sentiment using GPT
    user_sentiment = analyze_sentiment_gpt(user_message)

    # Initialize conversation history if not present
    if 'conversation_history' not in session:
        session['conversation_history'] = []
        session['interaction_count'] = 0  # Initialize interaction counter
        session['permanent'] = True  # Make session permanent

    # Increment interaction count
    session['interaction_count'] += 1

    # Add user message to conversation history
    session['conversation_history'].append({"role": "user", "content": user_message})

    # Extract the user's name if mentioned
    user_name = extract_user_name(session['conversation_history'])

    # Limit conversation history to prevent overflow
    MAX_HISTORY_LENGTH = 20
    if len(session['conversation_history']) > MAX_HISTORY_LENGTH:
        session['conversation_history'] = session['conversation_history'][-MAX_HISTORY_LENGTH:]

    system_content = (
        f"You are a compassionate and understanding companion. The user's current sentiment appears to be {user_sentiment}. "
        "Adjust your tone and approach based on this sentiment. "
        "For positive sentiment, be encouraging and celebratory. "
        "For negative sentiment, be empathetic and supportive. "
        "For neutral sentiment, be gently inquisitive and engaging. "
        "Always prioritize active listening and emotional support. Avoid giving medical advice or making diagnoses."
    )

    # Add language instruction
    language_instruction = f"Respond to the user in {preferred_language.capitalize()}."
    system_content += f" {language_instruction}"

    system_message = {"role": "system", "content": system_content}
    messages = [system_message] + session['conversation_history']

    assistant_message = generate_response(messages)
    session['conversation_history'].append({"role": "assistant", "content": assistant_message})

    # To do - begin
    # Add logic for prompt injection and content moderation
    # To do - end

    # Occasionally address the user by their name if a name was extracted to keep it more friendly
    if random.random() < 0.3 and user_name:
        assistant_message = f"{user_name}, {assistant_message}"

    # Ask about hobbies after every 5 interactions. Later change this part of the prompt - AP
    if session['interaction_count'] % 5 == 0:
        hobby_suggestion = (
            "By the way, I'd love to get to know you better. Do you have any hobbies or activities that you enjoy? "
            "Sometimes engaging in something we love can really help improve our mood."
        )
        assistant_message += f" {hobby_suggestion}"


    # Use the voice selected from the front-end for speech synthesis
    synthesize_speech(assistant_message, voice=selected_voice)

    # Return the assistant's name from the drop-down
    return jsonify({"response": assistant_message, "assistant_name": assistant_name})


# run Flask server
if __name__ == '__main__':
    app.run(debug=True)
