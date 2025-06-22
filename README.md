# Facial Expression Recognition using AI 🤖😊😡

This project is a **real-time facial expression recognition system** using **Convolutional Neural Networks (CNN)** and explainable AI techniques such as **LIME**. It detects human facial emotions through the webcam and provides visual explanations for the model’s predictions.

## 🚀 Features

- Real-time emotion detection via webcam using OpenCV.
- Predicts 7 emotions: `Angry`, `Disgust`, `Fear`, `Happy`, `Sad`, `Surprise`, and `Neutral`.
- Uses a trained CNN model (`emotion_model.h5`).
- Explainability integration using LIME to understand **why** the model made a certain prediction.

## 🖼️ Sample Emotions

| Angry | Happy | Sad |
|-------|-------|-----|
| ![angry](assets/angry.png) | ![happy](assets/happy.png) | ![sad](assets/sad.png) |

## 📁 Project Structure

├── app.py
├── model/
│ └── emotion_model.h5
├── utils/
│ ├── detect_face.py
│ └── explain_lime.py
├── requirements.txt
├── README.md
└── assets/ (optional for storing images)

r
Copy
Edit

## 🛠️ Requirements

Install all dependencies using pip:

```bash
pip install -r requirements.txt
Python Packages Used
opencv-python

numpy

tensorflow

keras

matplotlib

lime

scikit-learn

pillow

pandas

🔧 How to Run
Ensure your webcam is connected.

Make sure the emotion_model.h5 file is present inside the model/ directory.

Run the application:

bash
Copy
Edit
python app.py
The webcam window will pop up:

Press e to explain the detected face using LIME.

Press q to quit.

🧠 How It Works
The webcam captures frames.

Faces are detected using detect_face.py.

Each face is processed and passed through the pre-trained model.

The predicted emotion is displayed on the frame.

Pressing 'e' will use LIME to generate a heatmap showing what parts of the face contributed most to the prediction.

📌 Dependencies Explanation
OpenCV: Captures video and handles real-time display.

TensorFlow/Keras: Runs the pre-trained CNN model.

LIME: Provides model explanation for a selected face.

NumPy/Matplotlib: Data processing and visualization.

📦 Model Info
Trained on FER-2013 dataset or similar.

Accepts grayscale 48x48 face crops.

Normalized inputs (/255.0) before prediction.

📄 License
This project is open-source and available for academic or personal use. For commercial use, please contact the author.

✨ Credits
Made with ❤️ by Akshita Chahal
GitHub

