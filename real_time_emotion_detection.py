import cv2
from deepface import DeepFace

cap = cv2.VideoCapture(0)

print("Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    try:
        results = DeepFace.analyze(
            img_path=frame,
            actions=['emotion'],
            enforce_detection=False,
            detector_backend='opencv'
        )

        if not isinstance(results, list):
            results = [results]

        for result in results:
            region = result.get('region', {})
            if region:
                x, y, w, h = region.get('x', 0), region.get('y', 0), region.get('w', 100), region.get('h', 100)

                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

                emotions = result.get('emotion', {})
                if emotions:
                    sorted_emotions = sorted(emotions.items(), key=lambda item: item[1], reverse=True)

                    for i, (emotion, score) in enumerate(sorted_emotions):
                        text = f"{emotion}: {score:.2f}%"
                        cv2.putText(frame, text, (x, y + h + 20 + i * 20),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)

    except Exception as e:
        print(f"Detection error: {e}")

    cv2.imshow("Real-Time Emotion Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
