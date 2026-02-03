import cv2
import numpy as np
import pandas as pd
from datetime import datetime
import os
from scipy.spatial.distance import euclidean

# Load pre-trained face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Directory containing known face images
known_faces_dir = r'c:\Users\aritr\Downloads\Attendence with face\know_faces'

# Load known faces and extract embeddings
known_faces = []
known_names = []

def extract_face_embedding(image):
    """Extract face embedding using histogram"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    
    if len(faces) > 0:
        x, y, w, h = faces[0]
        face_crop = image[y:y+h, x:x+w]
        # Use histogram as simple embedding
        hist = cv2.calcHist([face_crop], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
        hist = cv2.normalize(hist, hist).flatten()
        return hist
    return None

# Load all known faces
if os.path.exists(known_faces_dir):
    for filename in os.listdir(known_faces_dir):
        if filename.endswith('.jpg') or filename.endswith('.png'):
            image = cv2.imread(f"{known_faces_dir}/{filename}")
            if image is not None:
                embedding = extract_face_embedding(image)
                if embedding is not None:
                    known_faces.append(embedding)
                    known_names.append(filename.split('.')[0])
    if len(known_faces) == 0:
        print(f"Warning: No faces found in {known_faces_dir}. Please add face images.")
else:
    print(f"Error: Directory {known_faces_dir} does not exist. Please create it and add face images.")

# Function to capture image
def capture_image():
    cam = cv2.VideoCapture(0)
    while True:
        ret, frame = cam.read()
        if not ret:
            print("Failed to grab frame")
            break
        cv2.imshow('Press Space to capture', frame)
        if cv2.waitKey(1) & 0xFF == ord(' '):
            break
    cam.release()
    cv2.destroyAllWindows()
    return frame if ret else None

# Function to recognize face
def recognize_face(captured_image):
    embedding = extract_face_embedding(captured_image)
    if embedding is None:
        print("No faces detected in the image.")
        return None
    
    if len(known_faces) == 0:
        print("No known faces loaded.")
        return None
    
    best_match = None
    best_distance = float('inf')
    
    for i, known_embedding in enumerate(known_faces):
        distance = euclidean(embedding, known_embedding)
        if distance < best_distance:
            best_distance = distance
            best_match = known_names[i]
    
    # Return match if distance is below threshold
    if best_distance < 1.5:
        return best_match
    return None

# Function to mark attendance
def mark_attendance(student_name, file='attendance.xlsx'):
    now = datetime.now()
    current_date = now.strftime("%Y-%m-%d")
    current_time = now.strftime("%H:%M:%S")
    try:
        df = pd.read_excel(file)
    except FileNotFoundError:
        df = pd.DataFrame(columns=["Name", "Date", "Time"])
    new_record_df = pd.DataFrame({"Name": [student_name], "Date": [current_date], "Time": [current_time]})
    df = pd.concat([df, new_record_df], ignore_index=True)
    df.to_excel(file, index=False)

# Main execution
def main():
    image = capture_image()
    if image is None:
        return
    student_name = recognize_face(image)
    if student_name is None:
        print("Student not recognized!")
        return
    mark_attendance(student_name)
    print(f"Attendance marked for {student_name}")

if __name__ == "__main__":
    main()
