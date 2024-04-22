import tkinter as tk
from flask import Flask, render_template, request, jsonify

import threading
import cv2
import tensorflow as tf
from PIL import Image, ImageTk
import os
import json
from fuzzywuzzy import fuzz
from gtts import gTTS
import speech_recognition as sr

app = Flask(__name__)
# Load the trained model
model = tf.keras.models.load_model("plant_classifier_model.h5")
all_plant_names = ['Arive-Dantu', 'Basale', 'Betel', 'Crape_Jasmine', 'Curry', 'Drumstick', 'Fenugreek', 'Guava',
                   'Hibiscus', 'Indian_Beech', 'Indian_Mustard', 'Jackfruit', 'Jamaica', 'Jamun', 'Jasmine',
                   'Karanda', 'Lemon', 'Mango', 'Mexican_Mint', 'Mint', 'Neem', 'Oleander', 'Parijata', 'Peepal',
                   'Pomegranate', 'Rasna', 'Rose_Apple', 'Roxburgh_fig', 'Sandalwood', 'Tulsi']
# Create a voice recognizer
recognizer = sr.Recognizer()

# Create the main application window
app = tk.Tk()
app.title("Plant Recognition and Description")

# Create labels to display the results
result_label = tk.Label(app, text="", font=("Helvetica", 16))
result_label.pack()

description_label = tk.Label(app, text="", font=("Helvetica", 12), wraplength=400, justify="left")
description_label.pack()

# Create a Boolean variable to track 'p' key press and image capture
capture_image_flag = False

def capture_image():
    global capture_image_flag
    capture_image_flag = True

def voice_command():
    with sr.Microphone() as source:
        print("Please say something...")
        recognizer.adjust_for_ambient_noise(source, duration=1)
        audio = recognizer.listen(source)

    try:
        # Recognize speech using Google Web Speech API
        query = recognizer.recognize_google(audio)
        print("You said: " + query)

        # Load plant information from JSON file
        with open('plants.json', 'r') as json_file:
            data = json.load(json_file)
            plants = data['plants']

        # Find the closest matching plant name
        best_match = None
        best_score = 0

        for plant in plants:
            plant_name = plant['name']
            score = fuzz.ratio(query.lower(), plant_name.lower())
            if score > best_score:
                best_match = plant_name
                best_score = score

        if best_match:
            # Display the predicted plant
            result_label.config(text=f"Predicted Plant: {best_match}")

            # Find the description of the recognized plant
            for plant in plants:
                if plant['name'] == best_match:
                    description = plant['description']
                    break
            else:
                description = "Description not available for this plant."

            description_label.config(text=f"Description: {description}")

            # Convert the description to speech using gTTS
            tts = gTTS(description)
            tts.save("plant_description.mp3")

            # Play the speech using the default audio player
            os.system("start plant_description.mp3")

        else:
            result_label.config(text="Plant not recognized.")

    except sr.UnknownValueError:
        print("Google Web Speech API could not understand audio.")
    except sr.RequestError as e:
        print(f"Could not request results from Google Web Speech API; {e}")

# Create buttons for capturing image and voice command
capture_button = tk.Button(app, text="Capture Image", command=capture_image)
capture_button.pack()

voice_button = tk.Button(app, text="Voice Command", command=voice_command)
voice_button.pack()

# Function to capture a real-time image and predict the plant
def predict_plant_image():
    global predicted_class

    # Use OpenCV to capture a real-time image from the camera
    capture = cv2.VideoCapture(0)  # 0 for the default camera (you can change this if needed)

    # Set the dimensions of the captured frame
    capture.set(3, 800)  # Width
    capture.set(4, 600)  # Height

    while True:
        ret, frame = capture.read()
        if not ret:
            result_label.config(text="Error capturing image from the camera.")
            break

        # Resize the captured image to 150x150 pixels
        resized_frame = cv2.resize(frame, (400, 300))

        preprocessed_image = resized_frame / 255.0

        # Ensure color channels are in the correct order (RGB)
        preprocessed_image = preprocessed_image[..., ::-1]

        # Convert the OpenCV image to a format that can be displayed in tkinter
        frame_rgb = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)
        frame_pil = Image.fromarray(frame_rgb)
        frame_tk = ImageTk.PhotoImage(image=frame_pil)

        # Update the camera label with the current frame
        camera_label.config(image=frame_tk)
        camera_label.image = frame_tk  # Keep a reference to avoid garbage collection

        # Check for user input to stop capturing
        key = cv2.waitKey(1)
        if key == ord("q"):
            break

        # Predict the plant when user presses 'p'
        elif key == ord("p") and capture_image_flag:
            # Save the captured image to a temporary file
            image_path = "captured_image.jpg"
            cv2.imwrite(image_path, resized_frame)

            # Load and preprocess the captured image for prediction
            img = tf.keras.preprocessing.image.load_img(image_path, target_size=(150, 150))
            img_array = tf.keras.preprocessing.image.img_to_array(img)
            img_array = tf.expand_dims(img_array, 0)  # Expand dimensions for the model

            # Predict the plant
            predictions = model.predict(img_array)
            predicted_class = all_plant_names[predictions.argmax()]

            # Display the predicted plant
            result_label.config(text=f"Predicted Plant: {predicted_class}")

            with open('plants.json', 'r') as json_file:
                data = json.load(json_file)
                plants = data['plants']

            # Find the description of the recognized plant
            for plant in plants:
                if plant['name'] == predicted_class:
                    description = plant['description']
                    break
            else:
                description = "Description not available for this plant."

            description_label.config(text=f"Description: {description}")

            # Convert the description to speech using gTTS
            tts = gTTS(description)
            tts.save("plant_description.mp3")

            # Play the speech using the default audio player
            os.system("start plant_description.mp3")

            # Reset the capture_image_flag
            capture_image_flag = False

    # Release the camera and close the OpenCV window
    capture.release()
    cv2.destroyAllWindows()

# Create a separate window for the camera capturing
camera_window = tk.Toplevel()
camera_window.title("Camera Feed")

# Create a label for the camera feed in the separate window
camera_label = tk.Label(camera_window)
camera_label.pack()

# Start capturing the camera feed in a separate thread
camera_thread = threading.Thread(target=predict_plant_image)
camera_thread.daemon = True
camera_thread.start()

# Start the GUI main loop
app.mainloop()
