from scipy.spatial import distance  # Used for calculating distances between points
from imutils import face_utils  # Provides utility functions for working with facial landmarks
from pygame import mixer  # Handles audio playback
import imutils
import dlib  # Provides face detection and facial landmark prediction
import cv2  # OpenCV, used for handling images and video
import numpy as np  # NumPy, for numerical operations

# Initialize audio mixer and load alert sound
mixer.init()
mixer.music.load("music.wav")

# Function to calculate Eye Aspect Ratio (EAR), which determines if the eye is closed
def eye_aspect_ratio(eye):
    # Calculate distances between the vertical eye landmarks
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    # Calculate distance between horizontal eye landmarks
    C = distance.euclidean(eye[0], eye[3])
    # Compute EAR
    ear = (A + B) / (2.0 * C)
    return ear

# Set parameters
frame_check = 30  # Consecutive frames to check for drowsiness
initial_frames = 50  # Initial frames for baseline EAR calibration
baseline_ear = None  # Placeholder for the average baseline EAR
personalized_thresh = 0.8  # Drowsiness threshold as a percentage of baseline EAR

# Initialize dlib’s face detector and facial landmark predictor
detect = dlib.get_frontal_face_detector()
predict = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Obtain indexes for left and right eye landmarks in the facial landmark array
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["right_eye"]

# Initialize video capture
cap = cv2.VideoCapture(0)  # Open the default camera
flag = 0  # Counter to track frames where EAR is below threshold
alert_playing = False  # Boolean to check if alert sound is playing
frame_count = 0  # Counter for initial frames used in baseline calibration
baseline_ear_list = []  # List to store EAR values for baseline calculation

while True:
    ret, frame = cap.read()  # Capture a frame from the camera
    if not ret:
        print("Failed to capture image")
        break

    # Flip frame horizontally for a mirror effect
    frame = cv2.flip(frame, 1)
    # Resize frame to a fixed width for consistency
    frame = imutils.resize(frame, width=450)
    # Convert frame to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect faces in the grayscale frame
    subjects = detect(gray, 0)
    for subject in subjects:
        # Predict facial landmarks for each face
        shape = predict(gray, subject)
        # Convert shape to NumPy array
        shape = face_utils.shape_to_np(shape)
    
        
        # Extract left and right eye coordinates
        leftEye = shape[lStart:lEnd]
        rightEye = shape[rStart:rEnd]
        
        # Calculate EAR for both eyes
        leftEAR = eye_aspect_ratio(leftEye)
        rightEAR = eye_aspect_ratio(rightEye)
        # Average EAR for both eyes
        ear = (leftEAR + rightEAR) / 2.0
        
        # Draw contours around eyes for visualization
        leftEyeHull = cv2.convexHull(leftEye)
        rightEyeHull = cv2.convexHull(rightEye)
        cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
        cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)

        # Capture baseline EAR in the initial frames
        if frame_count < initial_frames:
            baseline_ear_list.append(ear)
            frame_count += 1
            # Show calibration message on screen
            cv2.putText(frame, "Kalibrasi... Buka Mata Anda!", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        else:
            # Calculate baseline EAR once initial frames are processed
            if baseline_ear is None:
                baseline_ear = np.mean(baseline_ear_list)
                thresh = baseline_ear * personalized_thresh  # Set drowsiness threshold

            # Drowsiness detection
            if ear < thresh:
                flag += 1  # Increment counter if EAR below threshold
                if flag >= frame_check:
                    # Display drowsiness warning message
                    cv2.putText(frame, "   KAMU MENGANTUK   ", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    if not alert_playing:
                        # Play alert sound if not already playing
                        mixer.music.play(-1)
                        alert_playing = True
            else:
                # Reset counter and stop alert if user opens eyes
                flag = 0
                if alert_playing:
                    mixer.music.stop()
                    alert_playing = False

    # Display the resulting frame
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):  # Press 'q' to exit
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
