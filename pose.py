# pose.py
import json
import enum
import mediapipe as mp
import cv2
import numpy as np

mp_pose = mp.solutions.pose

class PoseLandmark(enum.IntEnum):
    """The 33 pose landmarks."""
    NOSE = 0
    LEFT_EYE_INNER = 1
    LEFT_EYE = 2
    LEFT_EYE_OUTER = 3
    RIGHT_EYE_INNER = 4
    RIGHT_EYE = 5
    RIGHT_EYE_OUTER = 6
    LEFT_EAR = 7
    RIGHT_EAR = 8
    MOUTH_LEFT = 9
    MOUTH_RIGHT = 10
    LEFT_SHOULDER = 11
    RIGHT_SHOULDER = 12
    LEFT_ELBOW = 13
    RIGHT_ELBOW = 14
    LEFT_WRIST = 15
    RIGHT_WRIST = 16
    LEFT_PINKY = 17
    RIGHT_PINKY = 18
    LEFT_INDEX = 19
    RIGHT_INDEX = 20
    LEFT_THUMB = 21
    RIGHT_THUMB = 22
    LEFT_HIP = 23
    RIGHT_HIP = 24
    LEFT_KNEE = 25
    RIGHT_KNEE = 26
    LEFT_ANKLE = 27
    RIGHT_ANKLE = 28
    LEFT_HEEL = 29
    RIGHT_HEEL = 30
    LEFT_FOOT_INDEX = 31
    RIGHT_FOOT_INDEX = 32
    
# Custom connections inspired by OpenPose's BODY_PARTS_KPT_IDS
BODY_PARTS_CONNECTIONS = [
    [0, 1], [0, 4], [1, 2], [2, 3], [4, 5], [5, 6], [3, 7], [6, 8], [9, 10],  # Head
    [11, 12], [11, 13], [13, 15], [12, 14], [14, 16], [11, 23], [12, 24],     # Upper Body
    [23, 24], [23, 25], [25, 27], [24, 26], [26, 28],                         # Lower Body
    [15, 17], [15, 19], [15, 21], [16, 18], [16, 20], [16, 22],              # Hands
    [27, 29], [29, 31], [28, 30], [30, 32]                                   # Feet
]
    
def extract_keypoints(results, frame_id):
    """Extract keypoints from MediaPipe results."""
    keypoints = []
    if results.pose_landmarks:
        for index, landmark in enumerate(results.pose_landmarks.landmark):
            keypoints.append({
                "frame_id": frame_id,
                "kpt_id": index,
                "name": PoseLandmark(index).name,  # Use enum to get the name
                'x': landmark.x,
                'y': landmark.y,
                'z': landmark.z,
                'visibility': landmark.visibility,
            })
    return keypoints

def save_keypoints_to_json(keypoints, filename):
    """Save keypoints to a JSON file."""
    with open(filename, 'w') as f:
        json.dump(keypoints, f, indent=4)

def save_summart_report_to_json(report, filename):
    """Save the summary report to a JSON file."""
    with open(filename, 'w') as f:
        json.dump(report, f, indent=4)

def process_pose(image):
    """Process the pose detection and return keypoints."""
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_rgb.flags.writeable = False
        results = pose.process(image_rgb)
        image_rgb.flags.writeable = True
        return results

def console_log(img, msg):
    font = cv2.FONT_HERSHEY_SIMPLEX
    fontScale = 0.5
    org = (10, 30)  # Coordinates for the top-left corner
    fontColor = (0, 255, 0)
    backgroundColor = (0, 0, 255)  # Red background color
    padding = 5
    lineType = 2  # Thickness of the line

    y = org[1]
    for key, value in msg.items():
        line = f"{key}: {value}"
        cv2.putText(img, line, (org[0], y), font, fontScale, fontColor, lineType)
        y += 20  # Adjust this value to control the spacing between lines

    return img

def draw_skeleton(img, ref_frame_kpts):
    """Draw a semi-transparent grey skeleton based on keypoints."""
    overlay = img.copy()
    line_color = (128, 128, 128)  # Grey in BGR
    line_thickness = 60
    height, width = img.shape[:2]
    keypoints_dict = {kpt["kpt_id"]: (int(kpt["x"] * width), int(kpt["y"] * height)) 
                      for kpt in ref_frame_kpts if kpt["visibility"] > 0.5}

    for connection in BODY_PARTS_CONNECTIONS:
        idx1, idx2 = connection
        if idx1 in keypoints_dict and idx2 in keypoints_dict:
            pt1 = keypoints_dict[idx1]
            pt2 = keypoints_dict[idx2]
            cv2.line(overlay, pt1, pt2, line_color, line_thickness)
    
    alpha = 0.5
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)
    return img
    