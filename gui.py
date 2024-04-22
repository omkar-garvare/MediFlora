import tkinter as tk
import threading
import cv2
import tensorflow as tf
from PIL import Image, ImageTk
import os

from fuzzywuzzy import fuzz
from gtts import gTTS
import speech_recognition as sr

# Load the trained model
model = tf.keras.models.load_model("new_model.h5")

# Define class names
all_plant_names = ['Arive-Dantu', 'Basale', 'Betel', 'Crape_Jasmine', 'Curry', 'Drumstick', 'Fenugreek', 'Guava',
               'Hibiscus', 'Indian_Beech', 'Indian_Mustard', 'Jackfruit', 'Jamaica', 'Jamun', 'Jasmine',
               'Karanda', 'Lemon', 'Mango', 'Mexican_Mint', 'Mint', 'Neem', 'Oleander', 'Parijata', 'Peepal',
               'Pomegranate', 'Rasna', 'Rose_Apple', 'Roxburgh_fig', 'Sandalwood', 'Tulsi']

image_paths = {
    'Arive-Dantu': 'Medical/Arive-Dantu/AV-S-001.jpg',
    'Basale': 'Medical/Basale/BA-S-002.jpg',
    'Betel': 'Medical/Betel/PB-S-001.jpg',
    'Crape_Jasmine': 'Medical/Crape_Jasmine/TD-S-007.jpg',
    'Curry': 'Medical/Curry/MK-S-013.jpg',
    'Drumstick': 'Medical/Drumstick/MO-S-002.jpg',
    'Fenugreek': 'Medical/Fenugreek/TF-S-001.jpg',
    'Guava': 'Medical/Guava/PG-S-001.jpg',
    'Hibiscus': 'Medical/Hibiscus/HR-S-001.jpg',
    'Indian_Beech': 'Medical/Indian_Beech/PP-S-001.jpg',
    'Jackfruit': 'Medical/Jackfruit/AH-S-001.jpg',
    'Jamaica': 'Medical/Jamaica/MC-S-001.jpg',
    'Jamun': 'Medical/Jamun/SC-S-001.jpg',
    'Jasmine': 'Medical/Jasmine/J-S-007.jpg',
    'Karanda': 'Medical/Karanda/CC-S-001.jpg',
    'Lemon': 'Medical/Lemon/CL-S-001.jpg',
    'Mango': 'Medical/Mango/MI-S-001.jpg',
    'Mexican_Mint': 'Medical/Mexican_Mint/PA-S-001.jpg',
    'Mint': 'Medical/Mint/M-S-001.jpg',
    'Neem': 'Medical/Neem/AI-S-001.jpg',
    'Oleander': 'Medical/Oleander/NO-S-001.jpg',
    'Parijata': 'Medical/Parijata/NA-S-001.jpg',
    'Peepal': 'Medical/Peepal/FR-S-001.jpg',
    'Pomegranate': 'Medical/Pomegranate/PG-S-001.jpg',
    'Rasna': 'Medical/Rasna/AG-S-001.jpg',
    'Rose_Apple': 'Medical/Rose_Apple/SJ-S-001.jpg',
    'Roxburgh_fig': 'Medical/Roxburgh_fig/FA-S-001.jpg',
    'Sandalwood': 'Medical/Sandalwood/SA-S-001.jpg',
    'Tulsi': 'Medical/Tulsi/OT-S-001.jpg',


    # Add paths for other plant images here
}

# Define a dictionary with plant descriptions
plant_descriptions = {
    'Arive-Dantu': 'Arive-Dantu: Also known as Amarnath, this plant can be used as a food to eat when on diet or '
                   'looking for weight loss as it is rich in fiber, extremely low in calories, have traces of fats and '
                   'absolutely no cholesterol. It is used to help cure ulcers, diarrhea, swelling of mouth or throat '
                   'and high cholesterol. It also has chemicals that act as antioxidants.',
    'Basale': 'Basale: Basale has an anti-inflammatory activity and wound healing ability. It can be helpful as a '
              'first aid, and the leaves of this plant can be crushed and applied to burns, scalds and wounds to help '
              'in healing of the wounds.',
    'Betel': 'Betel: The leaves of Betel possess immense therapeutic potential, and are often used in helping to '
             'cure mood swings and even depression. They are also quite an effective way to improve digestive health '
             'as they effectively neutralise pH imbalances in the stomach. The leaves are also full of many '
             'anti-microbial agents that combat the bacteria in your mouth',
    'Crape_Jasmine': 'Crape_Jasmine: Jasmine is used in the curing of liver diseases, such as hepatits, '
                     'and in abdominal pain caused due to intense diarrhea, or dysentery. The smell of Jasmine flowers '
                     'can be used to improve mood, reduce stress levels, and also to reduce food cravings. Jasmine can '
                     'also be used to help in fighting skin diseases and speed up the process of wound healing.',
    'Curry': 'Curry: Curry leaves have immense nutritional value with low calories, and they help you fight '
             'nutritional deficiency of Vitamin A, Vitamin B, Vitamin C, Vitamin B2, calcium and iron. It aids in '
             'digestion and helps in the treatment of morning sickness, nausea, and diarrhea. The leaves of this plant '
             'have properties that help in lowering blood cholesterol levels. It can also be used to promote hair '
             'growth and decrease the side effects of chemotherapy and radiotherapy',
    'Drumstick': 'Drumstick: Drumstick contains high amounts of Vitamin C and antioxidants, which help you to build up '
                 'your immune system and fight against common infections such as common cold and flu. Bioactive '
                 'compounds in this plant help to relieve you from thickening of the arteries and lessens the chance '
                 'of developing high blood pressure. An due to a high amount of calcium, Drumstick helps in developing '
                 'strong and healthy bones.',
    'Fenugreek': 'Fenugreek: Commonly known as Methi in Indian households, Fenugreek is a plant with many medical '
                 'abilities. It is said that Fenugreek can aid in metabolic condition such as diabetes and in '
                 'regulating the blood sugar. Fenugreek has also been found to be as effective as antacid medications '
                 'for heartburn. Due to its high nutritional value and less calories, it is also a food item to help '
                 'prevent obesity.',
    'Guava': 'Guava: Aside from bearing a delicious taste, the fruit of the Guava tree is a rich source of Vitamin C '
             'and antioxidants. It is especially effective against preventing infections such as Gastrointestinal '
             'infections, Respiratory infections, Oral/dental infections and Skin infections. It can also aid in the '
             'treatment of Hypertension, Fever, Pain, Liver and Kidney problems. ',
    'Hibiscus': 'Hibiscus: The tea of the hibiscus flowers are quite prevalent and are used mainly to lower blood '
                'pressure and prevent Hypertension. It is also used to relieve dry coughs. Some studies suggest that '
                'the tea has an effect in relieving from fever, diabetes, gallbladder attacks and even cancer. The '
                'roots of this plant can also be used to prepare a tonic.',
    'Indian_Beech': 'Indian Beech: Popularly known as Karanja in India, the Indian Beech is a medicinal herb used '
                    'mainly for skin disorders. Karanja  oil is applied to the skin to manage boils, rashes and eczema '
                    'as well as heal wounds as it has antimicrobial properties. The oil can also be useful in '
                    'arthritis due to it’s anti-inflammatory activities.',
    'Mustard': 'Mustard: Mustard and its oil is widely used for the relief of joint pain, swelling, fever, coughs and '
               'colds. The mustard oil can be used as a massage oil, skin serum and for hair treatment. The oil can '
               'also be consumed, and as it is high in monounsaturated fatty acids, Mustard oil turns out to be a '
               'healthy choice for your heart. ',
    'Jackfruit': 'Jackfruit: Jackfruits are full with Carotenoids, the yellow pigments that give jackfruit it’s '
                 'characteristic colour. is high in Vitamin A, which helps in preventing heart diseases and eye '
                 'problems such as cataracts and macular degeneration and provides you with an excellent eyesight.',
    'Jamaica': 'Jamaica: The Jamaican Cherry plant have Anti-Diabetic properties which can potential '
                       'cure type 2 diabetes. Jamaican Cherry tea contains rich amounts of nitric oxide, '
                       'which relaxes blood vessels, reducing the chance of hypertension. Other than that, '
                       'it can help to relieve paint, prevent infections, boost immunity and promote digestive '
                       'health.',
    'Jamun': 'Jamun: The fruit extract of the Jamun plant is used in treating the common cold, cough and flu. The '
             'bark of this tree contain components like tannins and carbohydrates that can be used to fight '
             'dysentery. Jamun juice is used for treating sore throat problems and is also effective in the '
             'enlargement of the spleen',
    'Jasmine': 'Jasmine: Jasmine is used in the curing of liver diseases, such as hepatits, and in abdominal pain '
               'caused due to intense diarrhea, or dysentery. The smell of Jasmine flowers can be used to improve '
               'mood, reduce stress levels, and also to reduce food cravings. Jasmine can also be used to help in '
               'fighting skin diseases and speed up the process of wound healing.',
    'Karanda': 'Karanda: Karanda is especially used in treating problems regarding digestion. It is used to cure worm '
               'infestation, gastritis, dermatitis, splenomegaly and indigestion. It is also useful for respiratory '
               'infections such as cough, cold, asthama, and even tuberculosis.',
    'Lemon': 'Lemon: Lemons are an excellent source of Vitamin C and fiber, and therefore, it lowers the risk factors '
             'leading to heart diseases. Lemons are also known to prevent Kidney Stones as they have Citric acid that '
             'helps in preventing Kidney Stones. Lemon, with Vitamin C and citric acid helps in the absorption of '
             'iron.',
    'Mango': 'Mango: Known as King of Fruits by many, Mango is also packed with many medicinal properties. Mangoes '
             'have various Vitamins, such as Vitamin C, K, A, and minerals such as Potassium and Magnesium. Mangoes '
             'are also rich in anitoxidants, which can reduce the chances of Cancer. Mangoes are also known to '
             'promote digestive health and heart health too.',
    'Mexican_Mint': 'Mexican_Mint: Mexican Mint is a traditional remedy used to treat a variety of conditions. The '
                    'leaves are a major part used for medicinal purposes. Mexican mint helpsin curing respiratory '
                    'illness, such as cold, sore throat, congestions, runny nose, and also help in natural skincare.',
    'Mint': 'Mint: Mint is used usually in our daily lives to keep bad mouth odour at bay, but besides that, '
            'this plant also help in a variety of other functions such as relieving Indigestion, and upset stomach, '
            'and can also improve Irritable Bowel Syndrome (IBS). Mint is also full of nutrients such as Vitamin A, '
            'Iron, Manganese, Folate and Fiber.',
    'Neem': 'Neem: Prevalent in traditional remedies from a long time, Neem is considered as a boon for Mankind. It '
            'helps to cure many skin diseases such as Acne, fungal infections, dandruff, leprosy, and also nourishes '
            'and detoxifies the skin. It also boosts your immunity and act as an Insect and Mosquito Repellent. It '
            'helps to reduce joint paint as well and prevents Gastrointestinal Diseases',
    'Oleander': 'Oleander: The use of this plant should be done extremely carefully, and never without the '
                'supervision of a doctor, as it can be a deadly poison. Despite the danger, oleander seeds and leaves '
                'are used to make medicine. Oleander is used for heart conditions, asthma, epilepsy, cancer, leprosy, '
                'malaria, ringworm, indigestion, and venereal disease.',
    'Parijata': 'Parijata: Parijata plant is used for varying purposes. It shows anti-inflammatory and antipyretic ('
                'fever-reducing) properties which help in managing pain and fever. It is also used as a laxative, '
                'in rheumatism, skin ailments, and as a sedative. It is also said to provide relief from the symptoms '
                'of cough and cold. Drinking fresh Parijat leaves juice with honey helps to reduce the symptoms of '
                'fever.',
    'Peepal': 'Peepal: The bark of the Peeple tree, rich in vitamin K, is an effective complexion corrector and '
              'preserver. It also helps in various ailments such as Strengthening blood capillaries, minimising '
              'inflammation, Healing skin bruises faster, increasing skin resilience, treating pigmentation issues, '
              'wrinkles, dark circles, lightening surgery marks, scars, and stretch marks.',
    'Pomegranate': 'Pomegranate: Pomegranate has a variety of medical benefits. It is rich in antioxidants, '
                   'which reduce inflation, protect cells from damage and eventually lower the chances of Cancer. It '
                   'is also a great source of Vitamin C and an immunity booster. Pomegranate has also shown to stall '
                   'the progress of Alzheimer disease and protect memory.',
    'Rasna': 'Rasna: The Rasna plant or its oil helps to reduce bone and joint pain and reduce the symptoms of '
             'rheumatioid arthritis. It can also be used to cure cough and cold, release mucus in the respiratory '
             'system and clear them, eventually facilitates easy breathing. Rasna can also be applied to wounds to '
             'aid them in healing.',
    'Rose apple': 'Rose apple: Rose apple’s seed and leaves are used for treating asthma and fever. Rose apples '
                  'improve brain health and increase cognitive abilities. They are also effective against epilepsy, '
                  'smallpox, and inflammation in joints. They contain active and volatile compounds that have been '
                  'connected with having anti-microbial and anti-fungal effects. ',
    'Roxburgh fig': 'Roxburgh fig: Roxburgh fig is noted for its big and round leaves. Leaves are crushed and the '
                    'paste is applied on the wounds. They are also used in diarrhea and dysentery.',
    'Sandalwood': 'Sandalwood: Sandalwood is used for treating the common cold, cough, bronchitis, fever, and sore '
                  'mouth and throat. It is also used to treat urinary tract infections (UTIs), liver disease, '
                  'gallbladder problems, heatstroke, gonorrhea, headache, and conditions of the heart and blood '
                  'vessels (cardiovascular disease).',
    'Tulsi': 'Tulsi: Tulsi plant has the potential to cure a lot of ailments, and is used a lot in traditional '
             'remedies. Tulsi can help cure fever, to treat skin problems like acne, blackheads and premature ageing, '
             'to treat insect bites. Tulsi is also used to treat heart disease and fever, and respiratory problems. '
    # Add descriptions for other plants here
}

recognizer = sr.Recognizer()

# Create the main application window
app = tk.Tk()
app.title("                               Plant Recognition and Description")

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

        # Find the closest matching plant name
        best_match = None
        best_score = 0

        for plant_name in all_plant_names:
            score = fuzz.ratio(query.lower(), plant_name.lower())
            if score > best_score:
                best_match = plant_name
                best_score = score

        if best_match:
            # Display the predicted plant
            result_label.config(text=f"Predicted Plant: {best_match}")

            # Display the plant description if available
            description = plant_descriptions.get(best_match, "Description not available for this plant.")
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
        key = cv2.waitKey(0)
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

            # Display the plant description if available
            description = plant_descriptions.get(predicted_class, "Description not available for this plant.")
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