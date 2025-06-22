import cv2
import numpy as np
from keras.models import load_model
from utils.detect_face import get_face
from utils.explain_lime import explain_prediction

model = load_model('model/emotion_model.h5')
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

cap = cv2.VideoCapture(0)

print("Press 'e' to explain prediction using LIME")
print("Press 'q' to quit")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    faces, gray = get_face(frame)
    for (x, y, w, h) in faces:
        roi = gray[y:y+h, x:x+w]
        roi_resized = cv2.resize(roi, (48, 48)).reshape(1, 48, 48, 1) / 255.0

        prediction = model.predict(roi_resized)
        emotion = emotion_labels[np.argmax(prediction)]

        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        cv2.putText(frame, emotion, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

    cv2.imshow("Facial Expression Recognition", frame)

    key = cv2.waitKey(1)
    if key == ord('e') and len(faces) > 0:
        (x, y, w, h) = faces[0]
        face_img = frame[y:y+h, x:x+w]
        explain_prediction(face_img)
    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
