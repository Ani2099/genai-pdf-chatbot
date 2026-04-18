#from google.colab import drive
#drive.mount('/content/drive')

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

mnist = tf.keras.datasets.mnist

(x_train,y_train),(x_test,y_test)=mnist.load_data()


x_train=tf.keras.utils.normalize(x_train,axis=1)
x_test=tf.keras.utils.normalize(x_test,axis=1)

model=tf.keras.models.Sequential()

model.add(tf.keras.layers.Flatten(input_shape=(28,28)))
model.add(tf.keras.layers.Dense(128,activation='relu'))
model.add(tf.keras.layers.Dense(128,activation='relu'))
model.add(tf.keras.layers.Dense(10,activation='softmax'))


model.compile(optimizer='adam',loss='sparse_categorical_crossentropy', metrics=['accuracy'])

model.fit(x_train,y_train, epochs=40)

model.save('handwrittendigit.keras')

model=tf.keras.models.load_model('handwrittendigit.keras')
loss,accuracy=model.evaluate(x_test,y_test)

loss
accuracy

model=tf.keras.models.load_model('handwrittendigit.keras')
no=1
while os.path.isfile(f"digits/image{no}.png"):
  try:
    # Read the image using the current 'no' in the loop
    img=cv2.imread(f"digits/image{no}.png")
    if img is None:
      print(f"Error: Could not read image digits/image{no}.png. File might be missing or corrupted.")
      no = no + 1
      continue

    # Convert to grayscale if it's a color image
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Resize the image to 28x28 pixels, as the model expects this input size
    img = cv2.resize(img, (28, 28))

    print(f"Image {no} before inversion:")
    plt.imshow(img, cmap=plt.cm.binary)
    plt.title(f"Image {no} - Before Inversion")
    plt.show()

    # Invert the image if necessary (MNIST has white digits on black background)
    # If your image has black digits on white background, this step is crucial
    img = np.invert(img)

    print(f"Image {no} after inversion and before normalization:")
    plt.imshow(img, cmap=plt.cm.binary)
    plt.title(f"Image {no} - After Inversion")
    plt.show()

    # Convert to float32 for normalization
    img = img.astype(np.float32)

    # Normalize the image using the same method as x_train (axis=1 for L2 norm per row)
    # For a single 28x28 image, normalize along axis=1
    img = tf.keras.utils.normalize(img, axis=1)

    # Add a batch dimension (e.g., from (28, 28) to (1, 28, 28)) for model prediction
    img_batch = np.expand_dims(img, axis=0)

    prediction = model.predict(img_batch)
    print(f"Full prediction array: {prediction}")
    print(f"The digit is probably a {np.argmax(prediction)}")

  except Exception as e:
      print(f"An unexpected error occurred while processing image{no}.png: {e}")
  finally:
    pass
  no=no+1