import cv2
from deepface import DeepFace

# Initialize OpenCV video capture
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    try:

        analysis = DeepFace.analyze(frame, actions=['age', 'gender', 'race', 'emotion'], enforce_detection=True)

        for face in analysis:
            x, y, w, h = face['region']['x'], face['region']['y'], face['region']['w'], face['region']['h']

            # Draw bounding box
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # Display analysis
            text = f"{face['dominant_emotion']}, {face['age']} yrs, {face['dominant_race']}, {face['gender']}"
            cv2.putText(frame, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    except Exception as e:
        print("No face detected:", e)

    # Show the frame
    cv2.imshow('DeepFace Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()