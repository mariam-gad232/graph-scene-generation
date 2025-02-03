import cv2
from scene_graph_generation import build_model , model
from scene_graph_generation import extract_relationships, draw_relationships
import torch

cap = cv2.VideoCapture(0)  # Open webcam
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # Convert frame to model format
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    inputs = {
        "image": torch.as_tensor(image.astype("float32").transpose(2, 0, 1)),
        "height": frame.shape[0],
        "width": frame.shape[1]
    }
    
    # Run inference
    with torch.no_grad():
        outputs = model([inputs])[0]
    
    # Extract predictions
    relationships = extract_relationships(outputs)
    
    # Draw output
    draw_relationships(frame, outputs)

    # Display results
    cv2.putText(frame, ", ".join(relationships), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.imshow("Scene Graph Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to quit
        break

cap.release()
cv2.destroyAllWindows()
