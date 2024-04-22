import numpy as np
import tensorflow as tf

# Assuming you have previously loaded the test data using the load_data() function
from main import load_data

(train_images, train_labels), (test_images, test_labels) = load_data()

# Load the trained model (if not already loaded)
model = tf.keras.models.load_model("plant_classifier_model.h5")

# Preprocess the test images (if necessary)
test_images = test_images / 255.0  # If you haven't normalized them already

# Predict the labels for the test images
predictions = model.predict(test_images)
predicted_labels = np.argmax(predictions, axis=1)

# Calculate the overall accuracy
overall_accuracy = np.mean(predicted_labels == test_labels) * 100
print(f"Overall Accuracy: {overall_accuracy:.2f}%")