import cv2
import torch
import numpy as np

# Load MiDaS model
midas = torch.hub.load("intel-isl/MiDaS", "MiDaS")

# Set device to GPU if available, otherwise use CPU
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
midas = midas.to(device)
midas.eval()

# Initialize camera capture
cap = cv2.VideoCapture(0)

while True:
    # Capture frame from camera
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture frame")
        break

    # Resize frame to expected input size of MiDaS model
    frame_resized = cv2.resize(frame, (384, 384))

    # Convert frame to tensor and normalize pixel values
    frame_tensor = torch.from_numpy(frame_resized.astype(np.float32) / 255.).permute(2, 0, 1).unsqueeze(0).to(device)

    # Perform depth estimation
    with torch.no_grad():
        prediction = midas(frame_tensor)

    # Convert prediction to numpy array
    depth_map = prediction.squeeze().cpu().numpy()

    # Normalize depth map for visualization
    depth_map = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

    # Display depth map
    cv2.imshow("Depth Map", depth_map)

    # Display original frame
    cv2.imshow("Original Frame", frame)

    # Check for exit key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("System Stopped")
        break

# Release camera resources
cap.release()
cv2.destroyAllWindows()
