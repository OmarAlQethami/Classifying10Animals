import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing import image
import numpy as np

loaded_model = keras.models.load_model("/content/model0.h5")

class_indices_to_names = {0: 'Butterfly',
                          1: 'Cat',
                          2: 'Chicken',
                          3: 'Cow',
                          4: 'Dog',
                          5: 'Elephant',
                          6: 'Horse',
                          7: 'Sheep',
                          8: 'Spider',
                          9: 'Squirrel'}

img_path = "/content/ManualValidationPhotos/cow.jpeg"
img = image.load_img(img_path, target_size=(299, 299))
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)
img_array /= 255.0

predictions = loaded_model.predict(img_array)
predicted_class_index = np.argmax(predictions[0])
predicted_class_name = class_indices_to_names[predicted_class_index]
confidence = predictions[0][predicted_class_index]

print(f"Predicted Animal: {predicted_class_name}, Confidence: {confidence * 100:.2f}%")