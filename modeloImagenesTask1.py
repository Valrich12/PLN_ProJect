import keras 
from keras.models import load_model
loaded_model = load_model('content\my_saved_model')
# Obtener la firma predeterminada del modelo
inference_func = loaded_model.signatures["serving_default"]

#Categorizar una imagen de internet
from PIL import Image
import requests
from io import BytesIO
import cv2
import numpy as np
import tensorflow as tf

def categorizar(image_path):
  # Leer la imagen con OpenCV
  img = cv2.imread(image_path)

  # Preprocesar la imagen
  img = cv2.resize(img, (224, 224))
  img = tf.keras.applications.mobilenet_v2.preprocess_input(img)

  # Hacer una predicción
  prediccion = loaded_model.predict(np.array([img]))

  # Obtener la clase con la mayor probabilidad
  predicted_class = np.argmax(prediccion, axis=-1)

  return predicted_class[0]

#0 = NonViolent, 1 = Violent
path = r'train_images/train_images\NonViolent\D_MCiujW4AMQYyB.jpg'
prediccion = categorizar(path)
print(prediccion)