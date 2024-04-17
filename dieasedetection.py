import cv2
import numpy as np
import time

# Load the pre-trained model 
from keras.models import load_model
model = load_model('Apple.h5')  

# Define the labels for the classes
labels = ['Apple Scab', 'Apple Black Rot', 'Apple Cedar Dust', 'Healthy'] 

# Initialize disease counts
disease_counts = {label: 0 for label in labels}

def display_counts():
    # Print the disease counts
    print("Disease Counts:")
    for label, count in disease_counts.items():
        print(f"{label}: {count}")

def main():
    # Create a VideoCapture object
    cap = cv2.VideoCapture(0)

    # Initialize timer
    start_time = time.time()
    display_interval = 1.0  # Display counts every 1 second

    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()
        
        # Convert the frame to the format expected by the model
        resized_frame = cv2.resize(frame, (224, 224))  # Resize to match the input size of the model
        resized_frame = np.expand_dims(resized_frame, axis=0)  # Add batch dimension
        normalized_frame = resized_frame / 255.0  # Normalize pixel values to [0, 1]

        # Use the model to make predictions
        predictions = model.predict(normalized_frame)

        # Get the predicted class label
        predicted_label_index = np.argmax(predictions)
        predicted_label = labels[predicted_label_index]

        # Display the predicted label on the frame
        cv2.putText(frame, predicted_label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Update disease counts
        if predicted_label != 'Healthy':
            disease_counts[predicted_label] += 1

        # Display the resulting frame
        cv2.imshow('Leaf Disease Detection', frame)

        # Check if it's time to display counts
        if time.time() - start_time >= display_interval:
            display_counts()
            start_time = time.time()

        # Check for exit key
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("System Stopped")
            break

    # Release the capture
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
