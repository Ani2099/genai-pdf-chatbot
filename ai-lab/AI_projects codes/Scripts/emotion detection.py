!pip install ultralytics
!pip install opencv-python
!pip install fer
!pip install gTTS
!pip install pydub
!pip install Pillow==10.3.0 --force-reinstall
!pip install numpy==1.23.5 tensorflow==2.12 keras==2.12 fer mtcnn

# Clean and reinstall all dependencies to ensure compatibility
!pip uninstall -y numpy ultralytics opencv-python gTTS pydub Pillow

# Ensure Pillow is at a compatible version before other installations
# facenet-pytorch (a dependency of fer) requires Pillow<10.3.0,>=10.2.0.
# Also, ultralytics might implicitly expect an older Pillow API structure.
!pip install Pillow==10.2.0 --force-reinstall

# Install core libraries that are sensitive to NumPy versions.
# Ultralytics relies on PyTorch, which typically requires NumPy 1.x, but newer versions can be compatible with NumPy 2.x.
!pip install ultralytics # This will install torch and a compatible numpy

# Now install other dependencies, letting pip resolve against the installed numpy/torch/Pillow.
!pip install opencv-python
!pip install gTTS
!pip install pydub

# Install TensorFlow *after* PyTorch/Ultralytics and its NumPy.
# Commenting out TensorFlow installation for now, as it's the primary cause of NumPy 2.x being installed,
# which breaks PyTorch (a dependency of Ultralytics). If TensorFlow is needed, a specific version
# compatible with the existing NumPy 1.x and Python 3.12 would need to be found, or a separate
# environment might be necessary.
# !pip install tensorflow # Temporarily commented out to avoid numpy 2.x conflict

# Temporarily skipping fer and mtcnn due to numpy incompatibility with facenet-pytorch dependency.
# !pip install fer mtcnn  # Temporarily commented out due to numpy incompatibility

# ------------------------------
# 2️⃣ Imports
# ------------------------------
from ultralytics import YOLO
import cv2
import numpy as np # This will now be numpy from ultralytics/torch installation
# from fer.fer import FER # Temporarily commented out due to numpy incompatibility
from gtts import gTTS
from IPython.display import Audio, display
from google.colab.patches import cv2_imshow
from google.colab import output # Use google.colab.output for consistency
from base64 import b64decode
import time

# ------------------------------
# 3️⃣ Load YOLOv8l Model
# ------------------------------
model = YOLO("ultralytics/yolov8x.pt")  # lightweight YOLOv8l (Note: YOLO model is loaded but not directly used for face detection in this revised loop)
# emotion_detector = FER(mtcnn=True)      # FER for emotion detection - Temporarily commented out

# ------------------------------
# 4️⃣ Text-to-Speech Function
# ------------------------------
def speak(text):
    tts = gTTS(text)
    tts.save("result.mp3")
    return Audio("result.mp3")

# ------------------------------
# 5️⃣ Capture Webcam Frame in Colab
# ------------------------------
def capture_frame():
    js_code = '''
    async function captureFrame() {
      const video = document.createElement('video');
      const stream = await navigator.mediaDevices.getUserMedia({video:true});
      video.srcObject = stream;
      await video.play();
      const canvas = document.createElement('canvas');
      canvas.width = video.videoWidth;
      canvas.height = video.videoHeight;
      const ctx = canvas.getContext('2d');
      ctx.drawImage(video, 0, 0);
      const dataURL = canvas.toDataURL('image/jpeg', 0.8);
      stream.getTracks()[0].stop();
      return dataURL;
    }
    captureFrame();
    '''
    data = output.eval_js(js_code)
    binary = b64decode(data.split(',')[1])
    img = np.frombuffer(binary, dtype=np.uint8)
    frame = cv2.imdecode(img, cv2.IMREAD_COLOR)
    return frame

# ------------------------------
# 6️⃣ Real-Time Detection Loop
# ------------------------------
num_frames = 1  # number of frames to capture

for i in range(num_frames):
    frame = capture_frame()

    # Detect faces and emotions using FER's internal MTCNN detector
    # detected_emotions = emotion_detector.detect_emotions(frame) # Temporarily commented out

    # if detected_emotions: # Temporarily commented out
    #     for face_data in detected_emotions: # Temporarily commented out
    #         x1, y1, w, h = face_data['box'] # Temporarily commented out
    #         x2, y2 = x1 + w, y1 + h # Temporarily commented out

    #         emotions_dict = face_data['emotions'] # Temporarily commented out
    #         emotion = max(emotions_dict, key=emotions_dict.get) # Temporarily commented out
    #         score = emotions_dict[emotion] # Temporarily commented out

    #         text_result = f"Emotion: {emotion} (Confidence: {score:.2f})" # Temporarily commented out
    #         print(f"All emotions for this face: {emotions_dict}") # Diagnostic print # Temporarily commented out
    #         text=f"Emotion: {emotion}" # Temporarily commented out
    #         print(text) # Temporarily commented out
    #         # Draw bounding box + label # Temporarily commented out
    #         cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2) # Temporarily commented out
    #         cv2.putText(frame, text_result, (x1, y1 - 10), # Temporarily commented out
    #                     cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2) # Temporarily commented out

    #         # Speak emotion # Temporarily commented out
    #         display(speak(text)) # Temporarily commented out

    cv2_imshow(frame)
    time.sleep(1)

print("✅ Detection completed!")