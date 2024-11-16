import cv2
import numpy as np
import mediapipe as mp

# Initialize Mediapipe Pose module
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_draw = mp.solutions.drawing_utils
mp_pose_connections = mp_pose.POSE_CONNECTIONS  # Get pose landmark connections

# Start video capture on the camera
cap = cv2.VideoCapture(0)

show_black_screen = False  # Flag to toggle display

while True:
    success, img = cap.read()  # Read each frame from the video feed
    if not success:
        break  # Exit if the frame is not read properly

    # Convert the image to RGB for Mediapipe processing
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    result = pose.process(img_rgb)

    # Create a black screen with the same dimensions as the original frame
    black_screen = np.zeros_like(img)

    if result.pose_landmarks:
        # Draw pose landmarks on the original image
        mp_draw.draw_landmarks(img, result.pose_landmarks, mp_pose_connections,
                               mp_draw.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=4),
                               mp_draw.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2))

        # Check if a particular hand landmark is raised (e.g., right wrist)
        right_wrist = result.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST]
        nose = result.pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE]

        # Condition to switch to black screen (e.g., right wrist above the nose)
        if right_wrist.y < nose.y:
            show_black_screen = True  # Activate black screen mode

        # Draw connections between pose landmarks on the black screen
        mp_draw.draw_landmarks(black_screen, result.pose_landmarks, mp_pose_connections,
                               mp_draw.DrawingSpec(color=(255, 255, 255), thickness=2, circle_radius=5),
                               mp_draw.DrawingSpec(color=(255, 255, 255), thickness=2, circle_radius=2))

    # Display the appropriate screen
    if show_black_screen:
        cv2.imshow("Pose Landmarks", black_screen)
    else:
        cv2.imshow("Pose Landmarks", img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
    if cv2.getWindowProperty("Pose Landmarks", cv2.WND_PROP_VISIBLE) < 1:  # Window is closed
        break

cap.release()
cv2.destroyAllWindows()
