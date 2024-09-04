import os
import cv2
import numpy as np
import pandas as pd
import pygame
import threading
from tensorflow import keras
from keras.models import model_from_json, Sequential
from keras.utils import img_to_array
from transformers import BertForSequenceClassification, BertTokenizer

# Load emotion model
emotion_name = {0: 'happy', 1: 'neutral', 2: 'sad', 3: 'surprise'}

# Load json and create model
with open('/home/tanisha/Documents/SPRO/Emotion-Recognition--main/models/emotion_model1.json', 'r') as json_file:
    loaded_model_json = json_file.read()

classifier = model_from_json(loaded_model_json, custom_objects={'Sequential': Sequential})
classifier.load_weights("/home/tanisha/Documents/SPRO/Emotion-Recognition--main/models/emotion_model1.h5")

# Load face cascade
face_cascade = cv2.CascadeClassifier('/home/tanisha/Documents/SPRO/Emotion-Recognition--main/models/haarcascade_frontalface_default.xml')

def detect_emotion(image):
    img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(image=img_gray, scaleFactor=1.3, minNeighbors=5)
    emotion_output = "No face detected"

    for (x, y, w, h) in faces:
        cv2.rectangle(image, pt1=(x, y), pt2=(x + w, y + h), color=(255, 0, 0), thickness=2)
        roi_gray = img_gray[y:y + h, x:x + w]
        roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)
        
        if np.sum([roi_gray]) != 0:
            roi = roi_gray.astype('float') / 255.0
            roi = img_to_array(roi)
            roi = np.expand_dims(roi, axis=0)
            prediction = classifier.predict(roi)[0]
            
            # Adjust indices to remove the 'angry' emotion
            prediction_adjusted = np.delete(prediction, 0)
            
            maxindex = int(np.argmax(prediction_adjusted))
            emotion_output = emotion_name[maxindex]
            label_position = (x, y)
            cv2.putText(image, emotion_output, label_position, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    return image, emotion_output

def load_model_and_tokenizer():
    # Load the model and tokenizer
    model = BertForSequenceClassification.from_pretrained("/home/tanisha/Documents/SPRO/Song_DB/saved_modelv2")
    tokenizer = BertTokenizer.from_pretrained("/home/tanisha/Documents/SPRO/Song_DB/saved_modelv2")
    print("Model and tokenizer loaded from 'saved_modelv2' directory.")
    return model, tokenizer

def load_csv(file_path):
    # Load your CSV file
    return pd.read_csv(file_path)

def play_song(song_path, control):
    pygame.mixer.init()
    pygame.mixer.music.load(song_path)
    pygame.mixer.music.play()
    
    while not control['stop']:
        command = input("Enter 's' to stop: ")
        if command.lower() == 's':
            pygame.mixer.music.stop()
            control['stop'] = True
            print("Playback stopped.")
        else:
            print("Invalid input. Please enter 's' to stop.")

def predict_emotion(df, emotion_label, download_dir):
    df_emotion = df[df['label'] == emotion_label]
    if df_emotion.empty:
        print(f"No songs found for the label: {emotion_label}")
        return
    random_song = df_emotion.sample(1).iloc[0]
    track_name = random_song['track_name']
    song_path = os.path.join(download_dir, f"{track_name}.mp3")
    if os.path.exists(song_path):
        print(f"Emotion detected is {emotion_label}, so let's listen to '{track_name}' by {random_song['artists']}")
        control = {'stop': False}
        play_thread = threading.Thread(target=play_song, args=(song_path, control))
        play_thread.start()
        play_thread.join()  # Wait for the playback thread to finish
    else:
        print(f"The song '{track_name}' for emotion '{emotion_label}' is not available in the directory.")

def main():
    print("Real Time Face Emotion Detection and Music Recommendation Application")

    cap = cv2.VideoCapture(0)

    # Directory where downloaded songs are stored
    download_dir = "/home/tanisha/Documents/SPRO/Song_DB/downloaded_songsv2"
    csv_file_path = '/home/tanisha/Documents/SPRO/Song_DB/selected_tracks_info.csv'
    
    model, tokenizer = load_model_and_tokenizer()
    df = load_csv(csv_file_path)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame, emotion_output = detect_emotion(frame)
        cv2.imshow('Webcam Preview', frame)
        key = cv2.waitKey(1) & 0xFF
        
        # Press 'c' to capture the image and recognize emotion
        if key == ord('c'):
            print("Capturing image...")
            result_image_path = "result_image.png"
            cv2.imwrite(result_image_path, frame)

            # Map detected emotion to an integer label
            emotion_map = {'sad': 0, 'happy': 1, 'neutral': 3, 'surprise': 2}
            if emotion_output in emotion_map:
                emotion_label = emotion_map[emotion_output]
                predict_emotion(df, emotion_label, download_dir)
            else:
                print(f"Emotion '{emotion_output}' not recognized for song recommendation.")

        # Press 'q' to quit
        elif key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
