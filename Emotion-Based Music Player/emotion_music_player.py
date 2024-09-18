import cv2
import numpy as np
from keras.models import load_model
import os
import pygame

# Load the pre-trained face detection model
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Load the emotion detection model without loading the optimizer state
emotion_model = load_model('emotion_model.h5', compile=False)
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# Initialize Pygame mixer for playing music
pygame.mixer.init()

# Function to play music based on detected emotion
def play_music(emotion):
    # Define the path to your music folder
    music_folder = r'C:\Users\Vineet\projects\Emotion-Based Music Player\music'  # Replace with the actual path to your music folder
    music_file = ''
    
    # Select music based on the detected emotion
    if emotion == 'Happy':
        music_file = os.path.join(music_folder, 'happy_song.mp3')
    elif emotion == 'Sad':
        music_file = os.path.join(music_folder, 'sad_song.mp3')
    elif emotion == 'Angry':
        music_file = os.path.join(music_folder, 'angry_song.mp3')
    elif emotion == 'Neutral':
        music_file = os.path.join(music_folder, 'neutral_song.mp3')
    # Add more emotions and corresponding songs as needed

    # Play the selected music
    if music_file and os.path.exists(music_file):
        pygame.mixer.music.load(music_file)
        pygame.mixer.music.play()
    else:
        print(f"No music file found for the emotion: {emotion}")

# Start video capture
video_capture = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame
    ret, frame = video_capture.read()
    
    # Convert to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    
    # Draw rectangles around detected faces and predict emotions
    for (x, y, w, h) in faces:
        roi_gray = gray[y:y + h, x:x + w]  # Region of interest
        roi_gray = cv2.resize(roi_gray, (64, 64))  # Resize to 64x64 to match the model's expected input size
        roi_gray = roi_gray.astype('float32') / 255
        roi_gray = np.expand_dims(roi_gray, axis=0)
        roi_gray = np.expand_dims(roi_gray, axis=-1)

        # Predict emotion
        prediction = emotion_model.predict(roi_gray)
        max_index = int(np.argmax(prediction))
        emotion = emotion_labels[max_index]

        # Display the emotion label
        cv2.putText(frame, emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        # Play music based on the detected emotion
        play_music(emotion)
    
    # Display the resulting frame
    cv2.imshow('Emotion Detection', frame)
    
    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and close windows
video_capture.release()
cv2.destroyAllWindows()
pygame.mixer.quit()

