import os
import joblib
import numpy as np
import cv2
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier


dataset_path = "DevanagariHandwrittenCharacterDataset\Train"

def load_dataset(dataset_path):
    images = []
    labels = []

    # Iterate over each folder (digit)
    for folder_name in os.listdir(dataset_path):
        folder_path = os.path.join(dataset_path, folder_name)

        # Iterate over each image in the folder
        for filename in os.listdir(folder_path):
            if filename.endswith(".png"):
                image_path = os.path.join(folder_path, filename)
                label = int(folder_name)

                # Read the image and resize it to a fixed size (e.g., 28x28)
                image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
                image = cv2.resize(image, (28, 28))

                # Append the image and label to the lists
                images.append(image)
                labels.append(label)

    # Convert the lists to NumPy arrays
    images = np.array(images)
    labels = np.array(labels)

    return images, labels

# Load the dataset
images, labels = load_dataset(dataset_path)

# Split the dataset into training and testing sets (e.g., 80% for training, 20% for testing)
train_images, test_images, train_labels, test_labels = train_test_split(
    images, labels, test_size=0.2, random_state=42
)


# Flatten the image data from 2D to 1D
train_images = train_images.reshape(train_images.shape[0], -1)
test_images = test_images.reshape(test_images.shape[0], -1)

# Normalize the pixel values to the range [0, 1]
train_images = train_images / 255.0
test_images = test_images / 255.0


# Create an MLP classifier
model = MLPClassifier(hidden_layer_sizes=(256, 128), max_iter=500)

# Train the model
model.fit(train_images, train_labels)

model_path = "handwritten_hindi\model.pkl"
joblib.dump(model, model_path)

# Make predictions on the testing set
predictions = model.predict(test_images)

# Calculate the accuracy of the model
accuracy = accuracy_score(test_labels, predictions)
print("Accuracy:", accuracy)

# model.save('handwritten_hindi.model')
print("model is saved")