import cv2
import numpy as np
from keras.models import load_model
import os
import pygame

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
emotion_model = load_model('emotion_model.h5', compile=False)
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
pygame.mixer.init()
emotion_history = []

def record_emotion(emotion):
    emotion_history.append(emotion)
    with open("emotion_history.txt", "a") as file:
        file.write(f"{emotion}\n")

music_folder_config = {
    'Happy': 'path_to_happy_songs',
    'Sad': 'path_to_sad_songs',
    'Angry': 'path_to_angry_songs',
    'Neutral': 'path_to_neutral_songs',
}

def play_music(emotion):
    folder = music_folder_config.get(emotion, None)
    if folder and os.path.exists(folder):
        files = [f for f in os.listdir(folder) if f.endswith('.mp3')]
        if files:
            pygame.mixer.music.load(os.path.join(folder, files[0]))
            pygame.mixer.music.play()

def apply_visual_effect(frame, emotion):
    if emotion == 'Happy':
        frame[:, :, 1] = np.clip(frame[:, :, 1] + 50, 0, 255)
    elif emotion == 'Sad':
        frame[:, :, 0] = np.clip(frame[:, :, 0] + 50, 0, 255)
    elif emotion == 'Angry':
        frame[:, :, 2] = np.clip(frame[:, :, 2] + 50, 0, 255)
    return frame

video_capture = cv2.VideoCapture(0)
frame_width = int(video_capture.get(3))
frame_height = int(video_capture.get(4))
out = cv2.VideoWriter('output.avi', cv2.VideoWriter_fourcc(*'XVID'), 20.0, (frame_width, frame_height))

while True:
    ret, frame = video_capture.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    for (x, y, w, h) in faces:
        roi_gray = gray[y:y + h, x:x + w]
        roi_gray = cv2.resize(roi_gray, (64, 64))
        roi_gray = roi_gray.astype('float32') / 255
        roi_gray = np.expand_dims(roi_gray, axis=0)
        roi_gray = np.expand_dims(roi_gray, axis=-1)
        prediction = emotion_model.predict(roi_gray)
        max_index = int(np.argmax(prediction))
        emotion = emotion_labels[max_index]
        record_emotion(emotion)
        cv2.putText(frame, emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        frame = apply_visual_effect(frame, emotion)
        play_music(emotion)
    out.write(frame)
    cv2.imshow('Emotion Detection', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
out.release()
cv2.destroyAllWindows()
pygame.mixer.quit()
