# NikolAI- Emotion-Responsive Music Player

This project uses computer vision and deep learning to detect facial emotions in real time and recommend songs based on the detected emotion on Raspberry Pi.

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

Run pip install -r requirements.txt in your terminal.

## Files & Directory Structure:

SPRO/
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
├── app.py
└── result_image.png

## How to Run the Code:

python app.py

## Instructions

### Real-Time Face Emotion Detection

- The application will start the webcam and display a real-time video feed.
- The application uses a pre-trained model to detect emotions from facial expressions.

### User Interaction

- **Press 'c'**: Capture the current frame from the webcam, detect the emotion, and recommend a song based on the detected emotion.
- **Press 's'**: Stop the music and close the webcam.

### Song Recommendation

- When an emotion is detected, the application will look for a song labelled with the detected emotion in the CSV file `selected_tracks_info.csv`.
- If a song is found, it will be played automatically.
- **Stopping the Song**: During song playback, you can stop the song by entering 's' in the console.

## Notes

- Ensure the songs are named correctly and placed in the `downloaded_songsv2` directory.
- The CSV file `selected_tracks_info.csv` should have the necessary song details, including emotional labels.

## Troubleshooting

- **No face detected**: Ensure your face is visible and well-lit.
- **No songs found for the label**: Check if the CSV file has songs labelled correctly for the detected emotions.
- **Invalid input to stop the song**: Make sure to enter 's' correctly in the console to stop the song playback.

## Demonstration

![Screenshot from 2024-08-24 07-23-39](https://github.com/user-attachments/assets/765dddc7-1030-436f-b6eb-cb5b698969a2)

- You can view the demonstration of the project at this link:
- https://drive.google.com/drive/folders/1NeFac56Sh_IGwYZzLtY_tbCWnDKtAOdJ?usp=drive_link



