# https://colab.research.google.com/drive/1LpiYflALSJN6PNJkvAdJua7KXbec3ylS#scrollTo=3hMaqoefF5XW

# speech to speech

#!pip install google-generativeai SpeechRecognition gTTS pydub

import os
import google.generativeai as genai
from IPython.display import Audio, display
from gtts import gTTS
os.environ["GEMINI_API_KEY"] = "AIzaSyDRT8gQZjOobFuG5A5g2dW09fgrIxg_ecI"
genai.configure(api_key=os.environ["GEMINI_API_KEY"])
model = genai.GenerativeModel("gemini-2.5-flash")
print("✅ Gemini API configured")

from IPython.display import Javascript, display
from google.colab import output
from pydub import AudioSegment # Re-adding pydub import for audio conversion


# JavaScript for recording audio
display(Javascript('''
async function recordAudio(){
const stream = await navigator.mediaDevices.getUserMedia({audio:true});
const mediaRecorder = new MediaRecorder(stream);
let chunks = [];


mediaRecorder.ondataavailable = e => chunks.push(e.data);
mediaRecorder.start();


await new Promise(resolve => setTimeout(resolve, 5000));
mediaRecorder.stop();


await new Promise(resolve => mediaRecorder.onstop = resolve);
const blob = new Blob(chunks, {type:'audio/wav'});
const arrayBuffer = await blob.arrayBuffer();
return new Uint8Array(arrayBuffer);
}
'''))
print("🎤 Recording... Speak now")
audio = output.eval_js('recordAudio()')
# Convert the dictionary returned by eval_js into a bytes object
audio_bytes = bytes(audio[str(i)] for i in range(len(audio)))

raw_audio_temp_path = "raw_audio_from_js.tmp"
with open(raw_audio_temp_path, "wb") as f:
 f.write(audio_bytes)

# Convert the raw audio to a standard WAV format using pydub
audio_segment = AudioSegment.from_file(raw_audio_temp_path)
file_to_recognize = "input_converted.wav"
audio_segment.export(file_to_recognize, format="wav")


print(f"✅ Audio recorded and saved as {file_to_recognize}")

import speech_recognition as sr
r = sr.Recognizer()
with sr.AudioFile(file_to_recognize) as source:
    audio_data = r.record(source)
    try:
        text = r.recognize_google(audio_data, language="en-IN")
        print("🧑 You said:", text)
    except sr.UnknownValueError:
        print("❌ Speech Recognition could not understand audio. Please try again with clear speech.")
    except sr.RequestError as e:
        print(f"❌ Could not request results from Google Speech Recognition service; {e}")

response = model.generate_content(text)
reply = response.text
print("🤖 Gemini says:\n", reply)
tts = gTTS(reply, lang="en")
tts.save("reply.mp3")
print("🔊 Playing Gemini response")
display(Audio("reply.mp3", autoplay=True))