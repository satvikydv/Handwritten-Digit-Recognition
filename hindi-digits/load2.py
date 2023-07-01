import cv2
import joblib
import numpy as np
import os
import matplotlib.pyplot as plt


# Load the trained model
model_path = "handwritten_hindi\model.pkl"
model = joblib.load(model_path)

# Function to preprocess input image
def preprocess_image(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    image = cv2.resize(image, (28, 28))
    image = image.reshape(1, -1)
    image = image / 255.0
    return image

# Function to predict the digit
def predict_digit(image_path):
    image = preprocess_image(image_path)
    prediction = model.predict(image)
    return prediction[0]


image_directory = "digits"

image_number = 27
while os.path.isfile(f"digits\digit{image_number}.png"):
    try:
        image_name = "digit" + str(image_number)
        custom_image_path = os.path.join(image_directory, image_name + ".png")
        predicted_digit = predict_digit(custom_image_path)
        print("Predicted Digit:", predicted_digit)

        img = cv2.imread(f"digits\digit{image_number}.png")[:,:,0]
        img = np.invert(np.array([img]))
        plt.imshow(img[0], cmap=plt.cm.binary)
        plt.show()
    except:
        print("Error!")
    finally:
        image_number += 1