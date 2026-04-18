from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image
from tensorflow.keras.optimizers import RMSprop
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import cv2
import os

import zipfile
zip_ref= zipfile.ZipFile('/content/orangeapple.zip','r')
zip_ref.extractall('/content')
zip_ref.close()

img=image.load_img('/content/appleorange/appleorange/training/orange/orange1.jpg')

plt.imshow(img)

cv2.imread("/content/appleorange/appleorange/training/orange/orange1.jpg").shape

train=ImageDataGenerator(rescale=1/255)
validation=ImageDataGenerator(rescale=1/255)

train_dataset=train.flow_from_directory("/content/appleorange/appleorange/training",target_size=(200,200),batch_size=3,class_mode='binary')
validation_dataset=validation.flow_from_directory("/content/appleorange/appleorange/training",target_size=(200,200),batch_size=3,class_mode='binary')

train_dataset.class_indices

model = tf.keras.models.Sequential([tf.keras.layers.Conv2D(16,(3,3),activation='relu',input_shape=(200,200,3)),
                                   tf.keras.layers.MaxPool2D(2,2),

                                   tf.keras.layers.Conv2D(32,(3,3),activation='relu'),
                                   tf.keras.layers.MaxPool2D(2,2),

                                   tf.keras.layers.Conv2D(32,(3,3),activation='relu'),
                                   tf.keras.layers.MaxPool2D(2,2),

                                   tf.keras.layers.Flatten(),

                                   tf.keras.layers.Dense(512,activation='relu'),

                                   tf.keras.layers.Dense(1,activation='sigmoid')

                                   ])

model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

model.fit= model.fit(
    train_dataset,
    validation_data=validation_dataset,
    epochs=10
)

dir_path='/content/appleorange/appleorange/testing'

for i in os.listdir(dir_path):
  img=image.load_img(dir_path+'//'+i,target_size=(200,200))

  if img is None:
    print("Error: Could not read the image file.")
  else:
    img_array = np.array(img)
    test_img = cv2.resize(img_array,(200,200))
    test_input = test_img.reshape((1,200,200,3))
    plt.imshow(test_img)
    plt.show()

    val=model.predict(test_input)
    if val == 0:
      print("apple")
    else:
      print("orange")
