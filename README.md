# CVProject1
Model is in google drive as commented in the private chat, reason it's 180M

To execute the mode, you can tru something like below.

#TO use any new image for prediction 
import numpy as np
import joblib
from PIL import Image

# Load the saved KNN model
knn = joblib.load('knn_model.joblib')

# Load the new image and preprocess it
new_image = Image.open('new_image.jpg').convert('L')  # convert to grayscale
new_image = new_image.resize((28, 28))  # resize to 28x28
new_image = np.array(new_image).reshape(1, -1)  # flatten to a 784-dimensional vector
new_image = new_image.astype('float32') / 255  # scale pixel values to [0, 1]

# Use the KNN model to make a prediction
new_image_pred = knn.predict(new_image)

print(f"Prediction: {new_image_pred}")
