import logging
import os
import tempfile
import time
import wave
from io import BytesIO
import threading # Using threading for stop signal handling (basic)

import pyaudio
import streamlit as st
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, AIMessage
from langchain_openai import ChatOpenAI
from openai import OpenAI, APIError, AuthenticationError, RateLimitError
from pydub import AudioSegment

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables from .env file
load_dotenv()

# Set up OpenAI client
# Check if API key is available
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    st.error("OpenAI API key not found. Please set the OPENAI_API_KEY environment variable.")
    st.stop() # Stop the app if API key is missing

client = OpenAI(api_key=openai_api_key)

# Set page configuration
st.set_page_config(page_title="Voice Assistant", page_icon="🎤")
st.title("🎤 Voice Assistant")

# Initialize session state for conversation history and recording state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "recording" not in st.session_state:
    st.session_state.recording = False
if "stop_recording" not in st.session_state:
    st.session_state.stop_recording = False

# Initialize LLM
llm = ChatOpenAI(model="gpt-4", temperature=0.7)

# --- Audio Recording Functions ---

# Global variable to signal stopping (used by threading approach)
stop_event = threading.Event()

def record_audio(record_seconds=5):
    """
    Record audio for a fixed duration or until stop_event is set,
    using PyAudio and writing to a temporary WAV file.
    Returns the file path of the recorded audio.
    """
    temp_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    temp_file_name = temp_file.name
    temp_file.close() # Close so that wave can write to it

    # Initialize PyAudio and set up the WAV file
    audio = None
    stream = None
    wf = None

    try:
        audio = pyaudio.PyAudio()
        wf = wave.open(temp_file_name, "wb")
        wf.setnchannels(1)  # Mono
        wf.setsampwidth(audio.get_sample_size(pyaudio.paInt16))  # 16-bit samples
        wf.setframerate(16000)  # 16kHz sample rate

        # Open the stream
        stream = audio.open(format=pyaudio.paInt16,
                            channels=1,
                            rate=16000,
                            input=True,
                            frames_per_buffer=1024) # Increased buffer size

        st.session_state.recording = True
        st.session_state.stop_recording = False
        stop_event.clear() # Clear stop event for new recording

        st.info(f"Recording started. Speak now...")

        frames = []
        start_time = time.time()

        # Record in chunks until duration or stop signal
        while st.session_state.recording and (time.time() - start_time) < record_seconds and not stop_event.is_set():
             try:
                 data = stream.read(1024, exception_on_overflow=False) # Read audio data
                 frames.append(data)
             except IOError as e:
                 # Handle potential input overflow or other stream errors
                 logger.warning(f"IOError during stream read: {e}")
                 st.warning("Audio input issue detected, trying to continue...")
             time.sleep(0.01) # Small sleep to prevent busy waiting

        st.session_state.recording = False
        st.info("Recording finished.")

        # Write all recorded frames to the WAV file
        wf.writeframes(b''.join(frames))

        logger.info(f"Audio recorded and saved to {temp_file_name}")
        return temp_file_name

    except Exception as e:
        st.error(f"Error during audio recording: {e}")
        logger.error(f"Error during audio recording: {e}")
        return None
    finally:
        # Clean up audio resources
        if stream is not None and stream.is_active():
            stream.stop_stream()
        if stream is not None:
            stream.close()
        if audio is not None:
            audio.terminate()
        if wf is not None:
            wf.close()
        # Temporary file cleanup is handled later in the main flow


def stop_recording_action():
    """Sets the stop recording flag."""
    st.session_state.stop_recording = True
    stop_event.set() # Set the stop event for threading approach
    logger.info("Stop recording requested.")


# Function to transcribe audio using Whisper
def transcribe_audio(audio_bytes):
    """Transcribes audio bytes using OpenAI's Whisper model."""
    audio_file_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
            f.write(audio_bytes)
            audio_file_path = f.name
            logger.info("WAV file created for transcription at: " + audio_file_path)

        with open(audio_file_path, "rb") as audio_file:
            transcript = client.audio.transcriptions.create(
                file=audio_file,
                model="whisper-1",
                response_format="text"
            )
        return transcript
    except FileNotFoundError:
         st.error("Temporary audio file not found for transcription.")
         logger.error("Temporary audio file not found for transcription.")
         return None
    except (APIError, AuthenticationError, RateLimitError) as e:
        st.error(f"OpenAI API error during transcription: {e}")
        logger.error(f"OpenAI API error during transcription: {e}")
        return None
    except Exception as e:
        st.error(f"Error transcribing audio: {e}")
        logger.error(f"Error transcribing audio: {e}")
        return None
    finally:
        # Clean up temporary file
        if audio_file_path and os.path.exists(audio_file_path):
            try:
                os.unlink(audio_file_path)
                logger.info(f"Deleted temporary file: {audio_file_path}")
            except Exception as e:
                logger.error(f"Failed to delete temporary file {audio_file_path}: {e}")


# Function to get response from ChatGPT
def get_chatgpt_response(transcription):
    """Gets a response from the ChatGPT model based on the transcription."""
    if not transcription or not transcription.strip():
        logger.warning("Empty transcription received")
        return "I couldn't understand what you said. Could you please speak again?"

    st.session_state.messages.append(HumanMessage(content=transcription))
    # Display user message immediately
    with st.chat_message("user"):
        st.write(transcription)

    # Get assistant response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                response = llm.invoke(st.session_state.messages)
                st.session_state.messages.append(AIMessage(content=response.content))
                st.write(response.content)
                return response.content
            except (APIError, AuthenticationError, RateLimitError) as e:
                st.error(f"OpenAI API error getting response: {e}")
                logger.error(f"OpenAI API error getting response: {e}")
                return "An error occurred while getting a response from the AI."
            except Exception as e:
                st.error(f"Error getting ChatGPT response: {e}")
                logger.error(f"Error getting ChatGPT response: {e}")
                return "An unexpected error occurred."


# Function to convert text to speech using OpenAI's API
def text_to_speech(text, voice="alloy"):
    """Converts text to speech using OpenAI's TTS API."""
    if not text:
        return None

    speech_file_path = None
    try:
        speech_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
        speech_file_path = speech_file.name
        speech_file.close()

        with st.spinner(f"Converting to speech with voice '{voice}'..."):
            response = client.audio.speech.create(
                model="tts-1",
                voice=voice, # Use selected voice
                input=text
            )
            response.stream_to_file(speech_file_path)

        audio_segment = AudioSegment.from_file(speech_file_path)
        logger.info(f"Speech generated and saved to {speech_file_path}")
        return audio_segment

    except FileNotFoundError:
         st.error("Temporary speech file not found.")
         logger.error("Temporary speech file not found.")
         return None
    except (APIError, AuthenticationError, RateLimitError) as e:
        st.error(f"OpenAI API error during text-to-speech: {e}")
        logger.error(f"OpenAI API error during text-to-speech: {e}")
        return None
    except Exception as e:
        st.error(f"Error generating speech: {e}")
        logger.error(f"Error generating speech: {e}")
        return None
    finally:
        # Clean up temporary file
        if speech_file_path and os.path.exists(speech_file_path):
            try:
                os.unlink(speech_file_path)
                logger.info(f"Deleted temporary file: {speech_file_path}")
            except Exception as e:
                logger.error(f"Failed to delete speech file {speech_file_path}: {e}")


# --- Main Application UI ---

def main():
    st.markdown("### Speak with your AI Assistant")
    st.markdown("Click **Record Audio** to capture your speech or type your message below. "
                "Once recorded/sent, your input will be processed by the AI.")

    # Input for recording duration
    record_duration = st.number_input("Recording duration (seconds)", min_value=1, max_value=60, value=5, step=1)

    # Select voice for TTS
    voice_options = ["alloy", "echo", "fable", "onyx", "nova", "shimmer"]
    selected_voice = st.selectbox("Select AI Voice", voice_options)

    # Buttons for recording and stopping
    col1, col2 = st.columns(2)
    with col1:
        # Disable record button while recording is in progress
        if st.button("Record Audio", disabled=st.session_state.recording):
            # Use session state to trigger recording logic in the main script flow
            st.session_state.start_recording = True
            st.session_state.record_duration = record_duration # Store duration in state
            st.session_state.selected_voice = selected_voice # Store voice in state
            st.session_state.text_input_sent = False # Reset text input state
            st.rerun() # Rerun to start recording

    with col2:
        # Disable stop button if not currently recording
        if st.button("Stop Recording", disabled=not st.session_state.recording):
            stop_recording_action()
            # No rerun needed here, the recording loop will detect the stop_event

    # Handle recording logic after rerun
    if st.session_state.get('start_recording', False):
        st.session_state.start_recording = False # Reset the trigger
        recorded_file_path = record_audio(record_seconds=st.session_state.record_duration)
        # After recording finishes (either by duration or stop button), process the audio
        if recorded_file_path and os.path.exists(recorded_file_path):
             try:
                 with open(recorded_file_path, "rb") as f:
                     audio_bytes = f.read()
                 # Removed 'caption' argument
                 st.audio(audio_bytes, format="audio/wav")

                 with st.spinner("Transcribing audio..."):
                     transcription = transcribe_audio(audio_bytes)

                 if transcription:
                     st.success(f"Transcription: {transcription}")
                     response = get_chatgpt_response(transcription)
                     if response:
                         audio_response = text_to_speech(response, voice=st.session_state.selected_voice)
                         if audio_response:
                             buf = BytesIO()
                             audio_response.export(buf, format="mp3")
                             buf.seek(0)
                             # Removed 'caption' argument
                             st.audio(buf, format="audio/mp3")
                 else:
                     st.warning("Could not transcribe the audio.")

             except Exception as e:
                 st.error(f"Error processing the recorded audio: {e}")
                 logger.error(f"Error processing the recorded audio: {e}")
             finally:
                 # Ensure recorded file is deleted
                 if recorded_file_path and os.path.exists(recorded_file_path):
                     try:
                         os.unlink(recorded_file_path)
                         logger.info(f"Deleted recorded audio file: {recorded_file_path}")
                     except Exception as e:
                         logger.error(f"Failed to delete recorded audio file {recorded_file_path}: {e}")
        elif st.session_state.stop_recording:
            st.info("Recording stopped by user.")
        else:
            st.warning("Recording failed or was too short.")


    # Text input option
    text_input = st.text_input("Or type your message here:", key="text_input")
    if text_input and not st.session_state.get('text_input_sent', False):
        st.session_state.text_input_sent = True # Prevent re-processing on rerun
        st.session_state.selected_voice = selected_voice # Store voice for text input
        response = get_chatgpt_response(text_input)
        if response:
            audio_response = text_to_speech(response, voice=st.session_state.selected_voice)
            if audio_response:
                buf = BytesIO()
                audio_response.export(buf, format="mp3")
                buf.seek(0)
                # Removed 'caption' argument
                st.audio(buf, format="audio/mp3")
        st.rerun() # Rerun to clear text input and update chat

    # Display conversation history
    st.markdown("---") # Separator
    st.markdown("### Conversation History")
    # Display messages in reverse order to show latest at the bottom
    for message in reversed(st.session_state.messages):
        if isinstance(message, HumanMessage):
            with st.chat_message("user"):
                st.write(message.content)
        elif isinstance(message, AIMessage):
            with st.chat_message("assistant"):
                st.write(message.content)


    # Button to clear the conversation
    if st.button("Clear Conversation"):
        st.session_state.messages = []
        st.rerun()


if __name__ == "__main__":
    main()
