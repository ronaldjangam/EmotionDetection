import cv2
import dlib
import numpy as np
from keras.models import load_model


emotion_model = load_model('model_v6_23.hdf5')


emotion_labels = ['Anger', 'Disgust', 'Fear', 'Happy', 'Sadness', 'Surprise', 'Neutral']


face_detector = dlib.get_frontal_face_detector()
landmark_predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')


cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break


    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


    faces = face_detector(gray)

    for face in faces:

        landmarks = landmark_predictor(gray, face)


        for n in range(0, 68):
            x, y = landmarks.part(n).x, landmarks.part(n).y
            cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)


        x1, y1, x2, y2 = face.left(), face.top(), face.right(), face.bottom()
        roi = gray[y1:y2, x1:x2]


        roi = cv2.resize(roi, (48, 48))


        roi = roi.astype('float32') / 255
        roi = np.expand_dims(roi, axis=-1)  
        roi = np.expand_dims(roi, axis=0) 


        emotion_prediction = emotion_model.predict(roi)
        emotion_index = np.argmax(emotion_prediction)


        emotion_text = emotion_labels[emotion_index]
        cv2.putText(frame, emotion_text, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)


    cv2.imshow('Emotion Detection', frame)


    if cv2.waitKey(1) & 0xFF == 27:
        break


cap.release()
cv2.destroyAllWindows()
