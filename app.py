import cv2
from deepface import DeepFace

cap = cv2.VideoCapture(0)
print("Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    try:
        results = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)
        if not isinstance(results, list):
            results = [results]

        for res in results:
            x, y, w, h = res['region']['x'], res['region']['y'], res['region']['w'], res['region']['h']
            emotions = res['emotion']
            sorted_emotions = sorted(emotions.items(), key=lambda x: x[1], reverse=True)

            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            for i, (emo, score) in enumerate(sorted_emotions[:3]):
                text = f"{emo}: {score:.2f}%"
                cv2.putText(frame, text, (x, y + h + 20 + i * 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

    except Exception as e:
        print(f"Detection error: {e}")

    cv2.imshow("Webcam Emotion Detector", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
