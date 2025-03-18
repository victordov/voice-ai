import logging
import os
import tempfile
import time
import wave
from io import BytesIO  # For in-memory buffer

import pyaudio
import streamlit as st
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, AIMessage
from langchain_openai import ChatOpenAI
from openai import OpenAI
from pydub import AudioSegment

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables from .env file
load_dotenv()

# Set up OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Set page configuration
st.set_page_config(page_title="Voice Assistant", page_icon="🎤")
st.title("🎤 Voice Assistant")

# Initialize session state for conversation history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Initialize LLM
llm = ChatOpenAI(model="gpt-4", temperature=0.7)


# Refactored recording process using PyAudio
def record_audio(record_seconds=5):
    """
    Record audio for a fixed duration using PyAudio and write it to a temporary WAV file.
    Returns the file path of the recorded audio.
    """
    temp_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    temp_file_name = temp_file.name
    temp_file.close()  # Close so that wave can write to it

    # Initialize PyAudio and set up the WAV file
    audio = pyaudio.PyAudio()
    wf = wave.open(temp_file_name, "wb")
    wf.setnchannels(1)  # Mono
    wf.setsampwidth(audio.get_sample_size(pyaudio.paInt16))  # 16-bit samples
    wf.setframerate(16000)  # 16kHz sample rate

    # Define a callback to write data to the WAV file
    def callback(in_data, frame_count, time_info, status):
        wf.writeframes(in_data)
        return (in_data, pyaudio.paContinue)

    # Open the stream with the callback
    stream = audio.open(format=pyaudio.paInt16,
                        channels=1,
                        rate=16000,
                        input=True,
                        frames_per_buffer=1024,
                        stream_callback=callback)

    stream.start_stream()
    st.info(f"Recording for {record_seconds} seconds...")
    time.sleep(record_seconds)
    stream.stop_stream()
    stream.close()
    audio.terminate()
    wf.close()
    logger.info(f"Audio recorded and saved to {temp_file_name}")
    return temp_file_name


# Function to transcribe audio using Whisper
def transcribe_audio(audio_bytes):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
        f.write(audio_bytes)
        audio_file_path = f.name
        logger.info("WAV file created for transcription at: " + audio_file_path)
    try:
        with open(audio_file_path, "rb") as audio_file:
            transcript = client.audio.transcriptions.create(
                file=audio_file,
                model="whisper-1",
                response_format="text"
            )
        return transcript
    except Exception as e:
        st.error(f"Error transcribing audio: {e}")
        logger.error(f"Error transcribing audio: {e}")
        return None
    finally:
        try:
            os.unlink(audio_file_path)
        except Exception as e:
            logger.error(f"Failed to delete temporary file: {e}")


# Function to get response from ChatGPT
def get_chatgpt_response(transcription):
    if not transcription or not transcription.strip():
        logger.warning("Empty transcription received")
        return "I couldn't understand what you said. Could you please speak again?"

    st.session_state.messages.append(HumanMessage(content=transcription))
    with st.chat_message("user"):
        st.write(transcription)
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = llm.invoke(st.session_state.messages)
            st.session_state.messages.append(AIMessage(content=response.content))
            st.write(response.content)
    return response.content


# Function to convert text to speech using OpenAI's API
def text_to_speech(text):
    with st.spinner("Converting to speech..."):
        speech_file_path = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3").name
        try:
            response = client.audio.speech.create(
                model="tts-1",
                voice="alloy",
                input=text
            )
            response.stream_to_file(speech_file_path)
            audio_segment = AudioSegment.from_file(speech_file_path)
            return audio_segment
        except Exception as e:
            st.error(f"Error generating speech: {e}")
            logger.error(f"Error generating speech: {e}")
            return None
        finally:
            try:
                os.unlink(speech_file_path)
            except Exception as e:
                logger.error(f"Failed to delete speech file: {e}")


# Main UI
def main():
    st.markdown("### Speak with your AI Assistant")
    st.markdown("Click **Record Audio (5 seconds)** to capture your speech. "
                "Once recorded, your audio will be transcribed, and then processed by ChatGPT.")

    # Display conversation history
    for message in st.session_state.messages:
        if isinstance(message, HumanMessage):
            with st.chat_message("user"):
                st.write(message.content)
        elif isinstance(message, AIMessage):
            with st.chat_message("assistant"):
                st.write(message.content)

    # Button to trigger audio recording
    if st.button("Record Audio (5 seconds)"):
        with st.spinner("Recording..."):
            recorded_file_path = record_audio(record_seconds=5)
        try:
            with open(recorded_file_path, "rb") as f:
                audio_bytes = f.read()
            st.audio(audio_bytes, format="audio/wav")

            with st.spinner("Transcribing audio..."):
                transcription = transcribe_audio(audio_bytes)
            if transcription:
                st.success(f"Transcription: {transcription}")
                response = get_chatgpt_response(transcription)
                if response:
                    audio_response = text_to_speech(response)
                    if audio_response:
                        buf = BytesIO()
                        audio_response.export(buf, format="mp3")
                        buf.seek(0)
                        st.audio(buf, format="audio/mp3")
        except Exception as e:
            st.error(f"Error processing the recorded audio: {e}")
            logger.error(f"Error processing the recorded audio: {e}")
        finally:
            try:
                os.unlink(recorded_file_path)
            except Exception as e:
                logger.error(f"Failed to delete the recorded audio file: {e}")

    # Button to clear the conversation
    if st.button("Clear Conversation"):
        st.session_state.messages = []
        st.experimental_rerun()


if __name__ == "__main__":
    main()
