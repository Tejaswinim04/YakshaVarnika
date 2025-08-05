import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np

# Load model
model = tf.keras.models.load_model('yakshagana_cnn_model.h5')

# Load and preprocess image
img_path = 'path_to_your_image.jpg'
img = image.load_img(img_path, target_size=(224,224))
img_array = image.img_to_array(img) / 255.0
img_array = np.expand_dims(img_array, axis=0)

# Predict
pred = model.predict(img_array)
class_index = np.argmax(pred)
confidence = np.max(pred)

# Get class labels
labels = list(train_generator.class_indices.keys())

print("Prediction:", labels[class_index])
print("Confidence:", confidence)
