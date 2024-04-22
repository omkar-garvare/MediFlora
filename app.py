import json
import cv2
import numpy as np
import tensorflow as tf
from flask import Flask, render_template, request, jsonify, send_file, Response
from fuzzywuzzy import fuzz
from gtts import gTTS
import speech_recognition as sr

# Load the trained model
model = tf.keras.models.load_model("model_sev.h5")
all_plant_names = ['Arive-Dantu', 'Basale', 'Betel', 'Crape_Jasmine', 'Curry', 'Drumstick', 'Fenugreek', 'Guava',
                       'Hibiscus', 'Indian_Beech', 'Indian_Mustard', 'Jackfruit', 'Jamaica', 'Jamun', 'Jasmine',
                   'Karanda', 'Lemon', 'Mango', 'Mexican_Mint', 'Mint', 'Neem', 'Oleander', 'Parijata', 'Peepal',
                   'Pomegranate', 'Rasna', 'Rose_Apple', 'Roxburgh_fig', 'Sandalwood', 'Tulsi']

# Create a voice recognizer
recognizer = sr.Recognizer()

# Create the Flask app
app = Flask(__name__)

# Variable to track 'p' key press and image capture5
capture_image_flag = False

# Load plant information from JSON file
with open('plants.json', 'r') as json_file:
    plant_data = json.load(json_file)

# Create a dictionary to store plant descriptions
plant_descriptions = {}

# Populate plant_descriptions from the loaded JSON data
for plant in plant_data['plants']:
    plant_name = plant['name']
    description = plant['description']
    plant_descriptions[plant_name] = description

# Create a function to capture and predict the plant from the camera
def generate_frames():
    global capture_image_flag
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open camera.")

    while True:
        success, frame = cap.read()
        if not success:
            break
        else:
            if capture_image_flag:
                # Preprocess the captured image for prediction
                img = cv2.resize(frame, (150, 150))
                img = img / 255.0
                img = img.reshape(1, 150, 150, 3)

                # Predict the plant
                predictions = model.predict(img)
                predicted_class = all_plant_names[predictions.argmax()]

                # Display the predicted plant on the frame
                cv2.putText(frame, f"Predicted Plant: {predicted_class}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()  # Release the camera when it's no longer needed
    cv2.destroyAllWindows()  # Close any OpenCV windows

def preprocess_image(frame):
    try:
        # Resize the captured image to 150x150 pixels
        resized_frame = cv2.resize(frame, (150, 150))

        # Normalize pixel values to be in the range [0, 1]
        preprocessed_image = resized_frame.astype(np.float32) / 255.0

        # Ensure color channels are in the correct order (RGB)
        preprocessed_image = preprocessed_image[..., ::-1]

        # Reshape the image to match the model's input shape (add batch dimension)
        preprocessed_image = np.expand_dims(preprocessed_image, axis=0)

        return preprocessed_image

    except Exception as e:
        raise Exception(f"Error preprocessing image: {str(e)}")
@app.route('/predict_image', methods=['POST'])
def predict_image():
    try:
        image = request.files['image']

        # Load and preprocess the image
        # Load and preprocess the image
        img = cv2.imdecode(np.fromstring(image.read(), np.uint8), cv2.IMREAD_COLOR)
        preprocessed_image = preprocess_image(img)

        # Make predictions using your model (replace 'your_model' with 'model')
        predictions = model.predict(preprocessed_image)
        predicted_class = all_plant_names[predictions.argmax()]

        # Get the plant description based on the predicted class
        description = plant_descriptions.get(predicted_class, "Description not available for this plant.")

        # Return the predicted class and description as JSON
        return jsonify({'predicted_plant': predicted_class, 'description': description})
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/')
def index():
    return render_template('index.html', capture_image_flag=capture_image_flag)

@app.route('/capture_image', methods=['POST'])
def capture_image():
    try:
        # Capture an image from the camera
        cap = cv2.VideoCapture(0)
        ret, frame = cap.read()
        if not ret:
            return jsonify({'message': 'Error capturing image.'})

        # Preprocess the captured image for prediction
        img = cv2.resize(frame, (150, 150))
        img = img / 255.0
        img = img.reshape(1, 150, 150, 3)

        # Predict the plant
        predictions = model.predict(img)
        predicted_class = all_plant_names[predictions.argmax()]

        cap.release()
        cv2.destroyAllWindows()

        return jsonify({'predicted_plant': predicted_class})

    except Exception as e:
        return jsonify({'message': f'Error: {str(e)}'})

@app.route('/voice_command', methods=['POST'])
def voice_command():
    try:
        with sr.Microphone() as source:
            print("Please say something...")
            recognizer.adjust_for_ambient_noise(source, duration=1)
            audio = recognizer.listen(source)

        # Recognize speech using Google Web Speech API
        query = recognizer.recognize_google(audio)
        print("You said:", query)

        # Find the closest matching plant name
        best_match = None
        best_score = 0

        for plant in plant_data['plants']:
            plant_name = plant['name']
            score = fuzz.ratio(query.lower(), plant_name.lower())
            if score > best_score:
                best_match = plant_name
                best_score = score

        if best_match:
            # Find the description of the recognized plant
            for plant in plant_data['plants']:
                if plant['name'] == best_match:
                    description = plant['description']
                    break
            else:
                description = "Description not available for this plant."

            # Convert the description to speech using gTTS
            tts = gTTS(description)
            tts.save("static/plant_description.mp3")

            return jsonify({
                'voice_command_result': f"Predicted Plant: {best_match}\nDescription: {description}",
                'tts_audio_url': '/get_tts_audio',
                'predicted_plant': best_match,
            })
        else:
            return jsonify({'voice_command_result': 'Plant not recognized.'})

    except sr.UnknownValueError:
        print("Google Web Speech API could not understand audio.")
        return jsonify({'voice_command_result': 'Speech recognition failed.'})
    except sr.RequestError as e:
        print(f"Could not request results from Google Web Speech API; {e}")
        return jsonify({'voice_command_result': 'Speech recognition request failed.'})

@app.route('/get_tts_audio')
def get_tts_audio():
    return send_file('static/plant_description.mp3', as_attachment=True)

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)
