#!pip install -q google-generativeai
#!pip install -q pillow

#!pip install gTTS

from gtts import gTTS

from IPython.display import display, Audio
import google.generativeai as genai
from google.colab import files
from PIL import Image
import matplotlib.pyplot as plt

genai.configure(api_key=" ")
model = genai.GenerativeModel("gemini-2.5-flash")

uploaded = files.upload()
image_path = list(uploaded.keys())[0]
image = Image.open(image_path)
prompt = """
1. Describe the image clearly.
2. create a detailed cinematic story inspired by it.
3. Suggest a beautiful bollywood song as a back ground music for this image.
"""
response = model.generate_content([prompt, image])
response1=""
print(type(response.text))
for i in response.text:
  if i.isalpha()==True or i==" " or i=="\n":
    response1=response1+i
response=response1
print(response)
tts = gTTS(response)
tts.save('hello.mp3')
audio = Audio('hello.mp3', autoplay=True)
display(audio)

display(image)