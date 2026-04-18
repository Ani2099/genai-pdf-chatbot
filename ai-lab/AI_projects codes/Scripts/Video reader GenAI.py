#!pip install gTTS
from gtts import gTTS
from IPython.display import display, Audio

# ==============================
# 1. Install Required Library
# ==============================
!pip install -q google-generativeai

# ==============================
# 2. Import & Configure API
# ==============================
import google.generativeai as genai
import time

# 👉 Put your Gemini API Key here
genai.configure(api_key=" ")

# ==============================
# 3. Upload Video File
# ==============================
from google.colab import files

print("Upload your video file:")
uploaded = files.upload()

video_path = list(uploaded.keys())[0]
print(f"Uploaded file: {video_path}")

# ==============================
# 4. Upload Video to Gemini
# ==============================
print("\nUploading video to Gemini...")
video_file = genai.upload_file(path=video_path)

# Wait until processing is complete
while video_file.state.name == "PROCESSING":
    print("Processing video...")
    time.sleep(5)
    video_file = genai.get_file(video_file.name)

print("✅ Video processing complete!")

# ==============================
# 5. Load Gemini 2.5 Flash Model
# ==============================
model = genai.GenerativeModel("gemini-2.5-flash")
# (Note: currently 2.5-flash is used in SDK; 2.5 Flash naming may vary by release)

# ==============================
# 6. Video Reader Prompt
# ==============================
prompt = """
Analyze this video and provide:
1. Detailed description
2. Key events timeline
3. Objects and people detected
4. If speech exists, summarize it
"""

# ==============================
# 7. Generate Response
# ==============================
response = model.generate_content([video_file, prompt])

print("\n🎯 VIDEO ANALYSIS RESULT:\n")
print(response.text)

# ==============================
# 8. (Optional) Ask Questions
# ==============================
while True:
    q = input("\nAsk something about the video (or type 'exit'): ")
    if q.lower() == "exit":
        break

    try:
        ans = model.generate_content([video_file, q])
        print("\nAnswer:\n", A.text)
        tts = gTTS(ans.text) # Changed to ans.text to speak the answer to the question
        tts.save('hello.mp3')
        audio = Audio('hello.mp3', autoplay=True)
        display(audio)
    except Exception as e:
        print(f"\nError generating response or audio: {e}")
        print("Please try your question again or check your API key/quota.")

display(video_file)