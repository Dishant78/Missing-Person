import cv2
import numpy as np
from deepface import DeepFace
import mediapipe as mp

mp_face_detection = mp.solutions.face_detection
face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.5)

missing_persons_db = {
    "John Doe": "missing_faces/john_doe.jpg",
    "Jane Smith": "missing_faces/jane_smith.jpg"
}

def recognize_face(face_img):
    try:
        for name, img_path in missing_persons_db.items():
            result = DeepFace.verify(face_img, img_path, model_name="VGG-Face", enforce_detection=False)
            if result["verified"]:
                return name  
    except Exception as e:
        print(f"Error in face recognition: {e}")
    return None 

cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = face_detection.process(rgb_frame)

    if results.detections:
        for detection in results.detections:
            bboxC = detection.location_data.relative_bounding_box
            h, w, _ = frame.shape
            x, y, width, height = int(bboxC.xmin * w), int(bboxC.ymin * h), int(bboxC.width * w), int(bboxC.height * h)
            
            face_img = frame[y:y+height, x:x+width]

            if face_img.size != 0:
                person_name = recognize_face(face_img)
                if person_name:
                    label = f"Missing: {person_name}"
                    color = (0, 0, 255)  
                else:
                    label = "Unknown Face"
                    color = (0, 255, 0)  

                
                cv2.rectangle(frame, (x, y), (x + width, y + height), color, 2)
                cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

   
    cv2.imshow("Missing Person Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
