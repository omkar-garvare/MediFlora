import os
import pickle
import numpy as np
import cv2
import tensorflow as tf
from tqdm import tqdm
from keras.applications import VGG19
from keras.applications.vgg19 import preprocess_input
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from keras.callbacks import LearningRateScheduler


# Define class names and labels
class_names = ['Arive-Dantu', 'Basale', 'Betel', 'Crape_Jasmine', 'Curry', 'Drumstick', 'Fenugreek', 'Guava',
               'Hibiscus', 'Indian_Beech', 'Indian_Mustard', 'Jackfruit', 'Jamaica', 'Jamun', 'Jasmine',
               'Karanda', 'Lemon', 'Mango', 'Mexican_Mint', 'Mint', 'Neem', 'Oleander', 'Parijata', 'Peepal',
               'Pomegranate', 'Rasna', 'Rose_Apple', 'Roxburgh_fig', 'Sandalwood', 'Tulsi']
class_names_label = {class_name: i for i, class_name in enumerate(class_names)}
nb_classes = len(class_names)

# Function to preprocess an image
def pre_process(img_path, target_size=(150, 150)):
    image = cv2.imread(img_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, target_size)
    return image

# Function to load and preprocess the dataset
def load_data(data_dir, target_size=(150, 150), split_ratio=0.8):
    train_images = []
    train_labels = []
    test_images = []
    test_labels = []

    for class_name in class_names:
        class_dir = os.path.join(data_dir, class_name)

        for file in tqdm(os.listdir(class_dir)):
            img_path = os.path.join(class_dir, file)
            image = pre_process(img_path, target_size)

            if np.random.rand(1) < split_ratio:
                train_images.append(image)
                train_labels.append(class_names_label[class_name])
            else:
                test_images.append(image)
                test_labels.append(class_names_label[class_name])

    train_images = np.array(train_images, dtype='float32')
    train_labels = np.array(train_labels, dtype='int32')
    test_images = np.array(test_images, dtype='float32')
    test_labels = np.array(test_labels, dtype='int32')

    return (train_images, train_labels), (test_images, test_labels)

nb_trainable_classes = 25

# Load and preprocess the dataset
(train_images, train_labels), (test_images, test_labels) = load_data('Medical', target_size=(224, 224))

# Normalize pixel values to the range [0, 1]
train_images = train_images / 255.0
test_images = test_images / 255.0

# Load the pre-trained VGG19 model
base_model = VGG19(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Freeze the layers of the pre-trained model
for layer in base_model.layers:
    layer.trainable = False

# Create your custom classifier
x = base_model.output
x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.Dense(128, activation='relu')(x)
predictions = tf.keras.layers.Dense(nb_classes, activation='softmax')(x)

# Create the full model by combining the VGG19 base and the custom classifier
model = tf.keras.Model(inputs=base_model.input, outputs=predictions)

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Implement early stopping
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3)

# Define a learning rate schedule
initial_learning_rate = 0.001
def lr_schedule(epoch):
    if epoch < 5:
        return initial_learning_rate
    else:
        return initial_learning_rate * tf.math.exp(0.1 * (5 - epoch))

lr_scheduler = LearningRateScheduler(lr_schedule)

# Create data generators with data augmentation
train_datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest')

train_generator = train_datagen.flow(train_images, train_labels, batch_size=32)

# Train the model with data augmentation and the learning rate schedule
history = model.fit(train_generator, epochs=10, validation_data=(test_images, test_labels),
                    callbacks=[early_stopping, lr_scheduler])

# After training, count the number of trained and non-trained images
total_trained_images = sum(1 for label in train_labels if label < nb_trainable_classes)
total_non_trained_images = len(train_labels) - total_trained_images

# Print the counts
print(f"Total Trained Images: {total_trained_images}")
print(f"Total Non-Trained Images: {total_non_trained_images}")

# Save the trained model
model.save("vgg19_model.h5")

# Save the training history
with open("training_history.pkl", "wb") as history_file:
    pickle.dump(history.history, history_file)

# Plot accuracy and loss
def plot_accuracy_loss(history):
    # Plot accuracy and loss here
    pass

# Call the plot_accuracy_loss function to display accuracy and loss plots
plot_accuracy_loss(history)