import os
import pickle
import numpy as np
import seaborn as sn

sn.set(font_scale=1.4)
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
import cv2
import tensorflow as tf
from tqdm import tqdm

# Now we create the class names and store them in the labels.

class_names = ['Arive-Dantu', 'Basale', 'Betel', 'Crape_Jasmine', 'Curry', 'Drumstick', 'Fenugreek', 'Guava',
               'Hibiscus', 'Indian_Beech', 'Indian_Mustard', 'Jackfruit', 'Jamaica_Cherry-Gasagase', 'Jamun', 'Jasmine',
               'Karanda', 'Lemon', 'Mango', 'Mexican_Mint', 'Mint', 'Neem', 'Oleander', 'Parijata', 'Peepal',
               'Pomegranate', 'Rasna', 'Rose_Apple', 'Roxburgh_fig', 'Sandalwood', 'Tulsi']
class_names_label = {class_name: i for i, class_name in enumerate(class_names)}
nb_classes = len(class_names)
IMAGE_SIZE = (150, 150)


def pre_process(img_path):
    image = cv2.imread(img_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, IMAGE_SIZE)
    return image


def load_data():
    datasets = ['Medical']
    train_images = []
    train_labels = []
    test_images = []
    test_labels = []

    # Iterate through training and test sets
    for dataset in datasets:

        # Iterate through each folder corresponding to a category
        for folder in os.listdir(dataset):
            label = class_names_label[folder]

            # Iterate through each image in our folder
            for file in tqdm(os.listdir(os.path.join(dataset, folder))):
                # Get the path name of the image
                img_path = os.path.join(os.path.join(dataset, folder), file)

                # Open and resize the img
                image = pre_process(img_path)

                # Split the data into training and testing
                if np.random.rand(1) < 0.8:  # You can adjust the split ratio
                    train_images.append(image)
                    train_labels.append(label)
                else:
                    test_images.append(image)
                    test_labels.append(label)

    train_images = np.array(train_images, dtype='float32')
    train_labels = np.array(train_labels, dtype='int32')
    test_images = np.array(test_images, dtype='float32')
    test_labels = np.array(test_labels, dtype='int32')

    return (train_images, train_labels), (test_images, test_labels)


(train_images, train_labels), (test_images, test_labels) = load_data()
train_images, train_labels = shuffle(train_images, train_labels, random_state=25)

n_train = train_labels.shape[0]
n_test = test_labels.shape[0]

print("Number of training examples: {}".format(n_train))
print("Number of testing examples: {}".format(n_test))
print("Each image is of size: {}".format(IMAGE_SIZE))

train_images = train_images / 255.0
test_images = test_images / 255.0


def display_examples(class_names, images, labels):
    fig = plt.figure(figsize=(10, 10))
    fig.suptitle("Some examples of images of the dataset", fontsize=16)
    for i in range(25):
        plt.subplot(5, 5, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(images[i], cmap=plt.cm.binary)
        plt.xlabel(class_names[labels[i]])
    plt.show()


# A function to simply display only one random image
def display_random_image(class_names, images, labels):
    index = np.random.randint(images.shape[0])
    plt.figure()
    plt.imshow(images[index])
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.title('Image #{} : '.format(index) + class_names[labels[index]])
    plt.show()


display_examples(class_names, train_images, train_labels)

model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation=tf.nn.relu),
    tf.keras.layers.Dense(30, activation=tf.nn.softmax)
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

datagen.fit(train_images)  # Fit the data generator on your training images

# Implement early stopping
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3)

# Train the model with data augmentation
history = model.fit(datagen.flow(train_images, train_labels, batch_size=128),
                    epochs=30, validation_data=(test_images, test_labels), callbacks=[early_stopping])

model.save("plant_classifier_model.h5")
with open("training_history.pkl", "wb") as history_file:
    pickle.dump(history.history, history_file)


def plot_accuracy_loss(history):
    """
    We plot the accuracy and the loss during the training of the nn.
    """
    fig = plt.figure(figsize=(10, 5))

    # Plot accuracy
    plt.subplot(2, 2, 1)
    plt.plot(history.history['accuracy'], 'bo--', label="accuracy")
    plt.plot(history.history['val_accuracy'], 'ro--', label="val_accuracy")
    plt.title("train_acc vs val_acc")
    plt.ylabel("accuracy")
    plt.xlabel("epochs")
    plt.legend()

    # Plot loss function
    plt.subplot(2, 2, 2)
    plt.plot(history.history['loss'], 'bo--', label="loss")
    plt.plot(history.history['val_loss'], 'ro--', label="val_loss")
    plt.title("train_loss vs val_loss")
    plt.ylabel("loss")
    plt.xlabel("epochs")

    plt.legend()

    # This section below adds two empty subplots
    plt.subplot(2, 2, 3)
    plt.subplot(2, 2, 4)

    plt.show()


# Call the plot_accuracy_loss function to display accuracy and loss plots
plot_accuracy_loss(history)

# Now, you can display the random image
predictions = model.predict(test_images)
pred_labels = np.argmax(predictions, axis=1)

display_random_image(class_names, test_images, pred_labels)
plt.show()
