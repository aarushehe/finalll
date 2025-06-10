import cv2
import mediapipe as mp
import time
import numpy as np
from deepface import DeepFace

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True)
mp_drawing = mp.solutions.drawing_utils

LEFT_EYE = [33, 159, 158, 133, 153, 144]
RIGHT_EYE = [362, 386, 385, 263, 380, 373]

def eye_aspect_ratio(landmarks, eye_indices, image_shape):
    h, w = image_shape
    p = [landmarks[i] for i in eye_indices]
    coords = [(int(pn.x * w), int(pn.y * h)) for pn in p]
    v1 = np.linalg.norm(np.array(coords[1]) - np.array(coords[5]))
    v2 = np.linalg.norm(np.array(coords[2]) - np.array(coords[4]))
    h1 = np.linalg.norm(np.array(coords[0]) - np.array(coords[3]))
    return (v1 + v2) / (2.0 * h1)

EAR_THRESHOLD = 0.22
blink_count = 0
blink_start = None

cap = cv2.VideoCapture(0)
print("Press 'q' to quit.")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)

    if results.multi_face_landmarks:
        face = results.multi_face_landmarks[0]

        left_ear = eye_aspect_ratio(face.landmark, LEFT_EYE, frame.shape[:2])
        right_ear = eye_aspect_ratio(face.landmark, RIGHT_EYE, frame.shape[:2])
        avg_ear = (left_ear + right_ear) / 2.0

        if avg_ear < EAR_THRESHOLD:
            if blink_start is None:
                blink_start = time.time()
        else:
            if blink_start is not None:
                blink_duration = time.time() - blink_start
                if 0.1 < blink_duration < 0.5:
                    blink_count += 1
                    print(f"[BLINK] Count: {blink_count}")
                blink_start = None

        for idx in LEFT_EYE + RIGHT_EYE:
            lm = face.landmark[idx]
            x, y = int(lm.x * frame.shape[1]), int(lm.y * frame.shape[0])
            cv2.circle(frame, (x, y), 2, (255, 0, 255), -1)

        cv2.putText(frame, f"EAR: {avg_ear:.2f}", (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(frame, f"Blinks: {blink_count}", (20, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    try:
        emotion_result = DeepFace.analyze(
            img_path=frame,
            actions=['emotion'],
            enforce_detection=False,
            detector_backend='opencv'
        )

        if not isinstance(emotion_result, list):
            emotion_result = [emotion_result]
        emotions = emotion_result[0].get("emotion", {})
        top_emotion = max(emotions, key=emotions.get)
        confidence = emotions[top_emotion]
        cv2.putText(frame, f"Emotion: {top_emotion} ({confidence:.1f}%)",
                    (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

    except Exception as e:
        print(f"[DeepFace error]: {e}")

    cv2.imshow("Blink + Emotion Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
