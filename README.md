# Real-Time Face Emotion Detection and Music Recommendation Application

This project uses computer vision and deep learning to detect facial emotions in real-time and recommend songs based on the detected emotion.

## Prerequisites

Before you begin, ensure you have met the following requirements:

- Python 3.x installed on your machine.
- The following Python libraries installed:
  - `numpy`
  - `opencv-python`
  - `pandas`
  - `pygame`
  - `tensorflow`
  - `keras`
  - `transformers`

Run pip install -r requirements.txt in your terminal

## Files & Directory Structure:

Specialization_Project/
│
├── models/
│   ├── emotion_model1.json
│   ├── emotion_model1.h5
│   ├── haarcascade_frontalface_default.xml
│
├── Song_DB/
│   ├── saved_modelv2/
│   ├── selected_tracks_info.csv
│   ├── downloaded_songsv2/
│
├── main_script.py
└── result_image.png

## How to Run the Code:

cd Specialization_Project
python app.py

## Instructions

### Real-Time Face Emotion Detection

- The application will start the webcam and display a real-time video feed.
- The application uses a pre-trained model to detect emotions from facial expressions.

### User Interaction

- **Press 'c'**: Capture the current frame from the webcam, detect the emotion, and recommend a song based on the detected emotion.
- **Press 'q'**: Quit the application and close the webcam.

### Song Recommendation

- When an emotion is detected, the application will look for a song labeled with the detected emotion in the CSV file `selected_tracks_info.csv`.
- If a song is found, it will be played automatically.
- **Stopping the Song**: During song playback, you can stop the song by entering 's' in the console.

## Notes

- Ensure the songs are named correctly and placed in the `downloaded_songsv2` directory.
- The CSV file `selected_tracks_info.csv` should have the necessary song details, including labels for emotions.

## Troubleshooting

- **No face detected**: Ensure your face is visible and well-lit.
- **No songs found for the label**: Check if the CSV file has songs labeled correctly for the detected emotions.
- **Invalid input to stop the song**: Make sure to enter 's' correctly in the console to stop the song playback.




