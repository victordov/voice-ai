# Voice AI
## Overview
This repository contains a Python-based web application built using [Streamlit](https://streamlit.io/). The app provides a user-friendly web interface that allows interaction, visualization, and real-time updates based on the methods defined in `app.py`.
The main purpose of this app is to simplify and accelerate interaction with data or AI models through a simple, yet effective UI provided by Streamlit.
## What Does the App Do?
The app performs the following key functionalities:

### ✔️ Audio Recording
- Easily record audio directly from your microphone through the interface. (default: 5 seconds recording).

### ✔️ Audio Playback and Feedback
- Immediate playback of your recorded voice for confirmation and review.

### ✔️ Speech-to-Text Transcription
- Converts recorded audio into text using OpenAI's Whisper model to generate accurate transcriptions, providing you instant textual feedback.

### ✔️ Real-Time AI Conversations
- Sends transcripts to OpenAI's GPT-4 model to generate meaningful and engaging conversational replies based on user inputs.

### ✔️ Text-to-Speech Replies
- Converts the AI-generated text responses back into clear audio messages via OpenAI’s text-to-speech service.

### ✔️ Conversation History
- Maintains an interactive conversation history, making previous conversations visible throughout your session for better context-awareness.
- Option to clear chat history with a dedicated button.

## Prerequisites
- Python >=3.12
- [Streamlit](https://streamlit.io/)
- IDE: [PyCharm](https://www.jetbrains.com/pycharm/download/) (recommended)

## Setting Up a Virtual Environment
A virtual environment allows for package isolation. Follow these steps to set up one:
``` bash
# Create virtual environment (Replace 'venv' with your preferred name)
python3 -m venv venv

# Activate the virtual environment
# On Unix or MacOS:
source venv/bin/activate

# On Windows:
venv\Scripts\activate
```
## Installing Dependencies
To install all the required dependencies defined in `requirements.txt`, run the following:
``` bash
pip install -r requirements.txt
```
_Ensure you have `requirements.txt` available. If not, create one by running_:
``` bash
pip freeze > requirements.txt
```
## Running the Application
To run the application from terminal:
``` bash
streamlit run app.py
```
## Debugging with PyCharm
You can debug your Streamlit app within PyCharm IDE by following these steps:
1. Open PyCharm and load your project directory.
2. Open the **Run/Debug Configuration** dialog:
    - Click on the dropdown next to the **Run** button and select **Edit Configurations...**.

3. Add a new **Python** configuration:
    - **Script**: Select the absolute path to your Streamlit executable located in your virtual environment. For example:
``` 
   /Users/your-username/dev/proj/env-name/bin/streamlit
```
- **Parameters**: Set the following parameters in the **Parameters** field:
``` 
   run app.py
```
- **Working directory**: Specify your application's project directory.
- **Python interpreter**: Make sure to select your project's virtual environment Python interpreter. (`venv/bin/python`)

1. Apply changes and close the configuration dialog.
2. Click the **Debug** button in PyCharm to start your Streamlit application in debug mode.

Example setup:
- **Script**:
``` 
  /YOUR_PATH_TO_STREAMLIT/bin/streamlit
```
- **Parameters**:
``` 
  run app.py
```
- **Interpreter**:
``` 
  /YOUR_PATH_TO_STREAMLIT/SpeakTrim/bin/python
```
Once running, the Streamlit application can be accessed from the browser using the URL provided by Streamlit (commonly `http://localhost:8501`).
## Additional Documentation & Support
- Streamlit Documentation: [https://docs.streamlit.io/](https://docs.streamlit.io/)
- PyCharm Debugging Documentation: [https://www.jetbrains.com/help/pycharm/debugging-code.html](https://www.jetbrains.com/help/pycharm/debugging-code.html)

## License
(Add your project's license info here if applicable.)
