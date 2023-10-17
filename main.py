"""
Project: Use VGG16 model to predict common diseases on cucumber plants

Target:
- Predict some common diseases of cucumber plants
- Provide knowledge about CNN networks in general and VGG16 model in particular

Limit:
- Types of diseases: Anthracnose, Bacterial Wilt, Belly Rot,
					 Downy Mildew, Fresh Cucumber, Fresh Leaf,
					 Gummy Stem Blight, Pythium Fruit Rot 

Technology used:
- Libraries: Keras, TensorFlow, Scikit-Learn, OpenCV, Numpy

Author: To Quy Thanh, Dang Sy Vinh, Nguyen Van Hiep
Email: Tothanh1feb3.quinn@gamil.com

Note: This is an initial version of the project and may be expanded in the future.
"""

import keras
from keras.preprocessing import image
import numpy as np
import os

# Load model
model = keras.models.load_model("models/savedmodels/version1.keras")  # Replace to path to your model

# Load and preprocess the image for prediction
img_path = 'data/Anthracnose/Anthracnose (1).jpg'
img = image.load_img(img_path, target_size=(224, 224))
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)  # Reshape to (1, 224, 224, 3) for a single image

y_pred = model.predict(img_array)
predicted_class = np.argmax(y_pred)


# Result
print("result: {}".format(os.listdir("data/")[predicted_class]))
