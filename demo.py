import cv2
import os
import mediapipe as mp
import numpy as np
import json
import argparse
import datetime
from pose import extract_keypoints, save_keypoints_to_json, process_pose  # Import your new functions

EXPORTS_DIR = "detection/exports"
os.makedirs(EXPORTS_DIR, exist_ok=True)

# Initialize MediaPipe Pose
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

def generate_filename(base_name):
    """Generate a unique filename based on the base name and current timestamp."""
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    return os.path.join(EXPORTS_DIR, f"{base_name}_{timestamp}.json")

def process_webcam():
    """Process webcam video feed."""
    # Start capturing video
    cap = cv2.VideoCapture(0)
    keypoints_list = []
    frame_id = 0

    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                print("Ignoring empty camera frame.")
                continue

            # Recolor the image
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            # Make detection
            results = pose.process(image)
            # Recolor back to BGR 
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            
            # # Extract landmarks
            # try:
            #     landmarks = results.pose_landmarks.landmark
            #     # print(landmarks)
            # except:
            #     pass
            
            # Extract landmarks
            keypoints = extract_keypoints(results, frame_id)
            keypoints_list.append(keypoints)  # Store keypoints
            
            # Render detections
            if results.pose_landmarks:
                mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                        mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2),
                                            mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
                                            )

            cv2.imshow('MediaPipe Pose - Webcam', image)
            if cv2.waitKey(5) & 0xFF == 27:  # Press 'Esc' to exit
                break
    
            frame_id += 1
            
    # Save keypoints to JSON file before exiting
    filename = generate_filename("webcam")
    save_keypoints_to_json(keypoints_list, filename)
    
    cap.release()
    cv2.destroyAllWindows()

def process_video(video_path):
    """Process video file."""
    cap = cv2.VideoCapture(video_path)
    keypoints_list = []
    base_name = os.path.splitext(os.path.basename(video_path))[0]  # Get the base name of the video file
    frame_id = 0

    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                print("End of video.")
                break

            # Process the frame
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            results = pose.process(image)
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            
            # Extract landmarks
            keypoints = extract_keypoints(results, frame_id)
            keypoints_list.append(keypoints)  # Store keypoints
            
            # Draw landmarks
            if results.pose_landmarks:
                mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

            cv2.imshow('MediaPipe Pose - Video', image)
            if cv2.waitKey(5) & 0xFF == 27:  # Press 'Esc' to exit
                break
            
            frame_id += 1

    # Save keypoints to JSON file before exiting
    filename = generate_filename(base_name)
    save_keypoints_to_json(keypoints_list, filename)
    
    cap.release()
    cv2.destroyAllWindows()

def process_image(image_path):
    """Process a single image."""
    image = cv2.imread(image_path)
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        # Convert the BGR image to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = pose.process(image_rgb)
        
        # Draw landmarks
        if results.pose_landmarks:
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        cv2.imshow('MediaPipe Pose - Image', image)
        cv2.waitKey(0)  # Press any key to close the window

    cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MediaPipe Pose Estimation Demo")
    parser.add_argument("--video", type=str, default="", help="Path to video file or camera id (0 for webcam)")
    parser.add_argument("--image", type=str, default="", help="Path to image file")

    args = parser.parse_args()

    if args.video:
        process_video(args.video)
    elif args.image:
        process_image(args.image)
    else:
        process_webcam()
    