import face_recognition
import cv2
import os
import numpy as np

KNOWN_FACES_DIR = "known_faces"
TOLERANCE = 0.6  
MODEL = "hog" 

known_face_encodings = []
known_face_names = []

for filename in os.listdir(KNOWN_FACES_DIR):
    image_path = os.path.join(KNOWN_FACES_DIR, filename)
    image = face_recognition.load_image_file(image_path)
    encoding = face_recognition.face_encodings(image)

    if len(encoding) > 0: 
        known_face_encodings.append(encoding[0])
        known_face_names.append(os.path.splitext(filename)[0])  

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

   
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
       
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding, TOLERANCE)
        name = "Unknown"

        if True in matches:
            match_index = matches.index(True)
            name = known_face_names[match_index]

      
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 3)
        cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    cv2.imshow('Face Recognition', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
