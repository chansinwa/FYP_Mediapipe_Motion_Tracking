# pose.py
import json
import enum
import mediapipe as mp
import cv2
import numpy as np
import math

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
    
def extract_keypoints(results, frame_id):
    """Extract keypoints from MediaPipe results."""
    keypoints = []
    if results.pose_landmarks:
        for index, landmark in enumerate(results.pose_landmarks.landmark):
            keypoints.append({
                "frame_id": frame_id,
                "keypoint_index": index,
                "name": PoseLandmark(index).name,  # Use enum to get the name
                'x': landmark.x,
                'y': landmark.y,
                'z': landmark.z,
                'visibility': landmark.visibility,
            })
    return keypoints

def calculate_angle(a, b, c):
    """Calculate the angle between three points."""
    if a is None or b is None or c is None:
        return None
    
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    a_length = np.linalg.norm(b - c)
    b_length = np.linalg.norm(a - c)
    c_length = np.linalg.norm(a - b)
    angle = np.arccos((a_length**2 + c_length**2 - b_length**2) / (2 * a_length * c_length))
    return np.degrees(angle)

def calculate_distance(point1, point2):
    """Calculate the Euclidean distance between two points."""
    return np.linalg.norm(np.array(point1) - np.array(point2))

def draw_text_with_outline(img, text, position, font_scale, thickness):
    ## Draw text with an outline effect (black outline (border)).
    cv2.putText(img, text, (position[0]-1, position[1]-1), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), thickness + 1)
    cv2.putText(img, text, (position[0]+1, position[1]-1), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), thickness + 1)
    cv2.putText(img, text, (position[0]-1, position[1]+1), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), thickness + 1)
    cv2.putText(img, text, (position[0]+1, position[1]+1), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), thickness + 1)

    ## Draw the red text on top
    cv2.putText(img, text, position, cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness)

def draw_interior_sector(img, center, pt_a, pt_b, body_part = None):
    x_axis_angle: float = 0
    end_angle: float = 0
    radius = 30
    arc_color = (0, 127, 255)
    sector_color = (0, 127, 255)
    
    angle_a = calculate_clockwise_angle_from_x_axis(center, pt_a)
    angle_b = calculate_clockwise_angle_from_x_axis(center, pt_b)
    
    if angle_b < angle_a:
        temp = angle_a
        angle_a = angle_b
        angle_b = temp
        
    if angle_b - angle_a < 180:
        x_axis_angle = angle_a
        end_angle = angle_b - angle_a
    else:
        x_axis_angle = angle_b
        end_angle = 360 - (angle_b - angle_a)
        
    ## Draw the arc border line (start andgle always 0)
    cv2.ellipse(img, center, (radius, radius), x_axis_angle, 0, end_angle, arc_color, 2)
    
    ## Draw the filled sector
    transparency = 0.4
    original_img = img.copy()
    cv2.ellipse(img, center, (radius, radius), x_axis_angle, 0, end_angle, sector_color, -1)
    cv2.addWeighted(img, transparency, original_img, 1 - transparency, 0, img)

def is_valid_point(point):
    """Check if a point is valid."""
    return point is not None and all(coord >= -1 for coord in point)

def draw_angles(img, results):
    """Draw joint angles on the image."""
    if not results.pose_landmarks:
        return img

    # Convert normalized coordinates to pixel coordinates
    h, w = img.shape[:2]
    landmarks = results.pose_landmarks.landmark

    # Define key landmarks
    RShoulder = [landmarks[PoseLandmark.RIGHT_SHOULDER].x * w, landmarks[PoseLandmark.RIGHT_SHOULDER].y * h]
    RElbow = [landmarks[PoseLandmark.RIGHT_ELBOW].x * w, landmarks[PoseLandmark.RIGHT_ELBOW].y * h]
    RWrist = [landmarks[PoseLandmark.RIGHT_WRIST].x * w, landmarks[PoseLandmark.RIGHT_WRIST].y * h]

    LShoulder = [landmarks[PoseLandmark.LEFT_SHOULDER].x * w, landmarks[PoseLandmark.LEFT_SHOULDER].y * h]
    LElbow = [landmarks[PoseLandmark.LEFT_ELBOW].x * w, landmarks[PoseLandmark.LEFT_ELBOW].y * h]
    LWrist = [landmarks[PoseLandmark.LEFT_WRIST].x * w, landmarks[PoseLandmark.LEFT_WRIST].y * h]

    # Approximate Neck as midpoint between shoulders
    # Neck = [(RShoulder[0] + LShoulder[0]) / 2, (RShoulder[1] + LShoulder[1]) / 2]

    RHip = [landmarks[PoseLandmark.RIGHT_HIP].x * w, landmarks[PoseLandmark.RIGHT_HIP].y * h]
    RAnkle = [landmarks[PoseLandmark.RIGHT_ANKLE].x * w, landmarks[PoseLandmark.RIGHT_ANKLE].y * h]

    LHip = [landmarks[PoseLandmark.LEFT_HIP].x * w, landmarks[PoseLandmark.LEFT_HIP].y * h]
    LAnkle = [landmarks[PoseLandmark.LEFT_ANKLE].x * w, landmarks[PoseLandmark.LEFT_ANKLE].y * h]

    # Calculate angles
    r_arm_angle = calculate_angle(RShoulder, RElbow, RWrist)
    l_arm_angle = calculate_angle(LShoulder, LElbow, LWrist)
    r_shoulder_angle = calculate_angle(LShoulder, RShoulder, RElbow)
    l_shoulder_angle = calculate_angle(RShoulder, LShoulder, LElbow)
    r_hip_angle = calculate_angle(RShoulder, RHip, RAnkle)
    l_hip_angle = calculate_angle(LShoulder, LHip, LAnkle)

    # Calculate dynamic font sizes based on distances
    distance_r_arm = calculate_distance(RElbow, RWrist)
    font_scale_r = min(max(distance_r_arm / 100, 0.5), 0.5)
    distance_l_arm = calculate_distance(LElbow, LWrist)
    font_scale_l = min(max(distance_l_arm / 100, 0.5), 0.5)
    distance_r_shoulder = calculate_distance(RShoulder, RElbow)
    font_scale_r_shoulder = min(max(distance_r_shoulder / 100, 0.5), 0.5)
    distance_l_shoulder = calculate_distance(LShoulder, LElbow)
    font_scale_l_shoulder = min(max(distance_l_shoulder / 100, 0.5), 0.5)
    distance_r_hip = calculate_distance(RHip, RAnkle)
    font_scale_r_hip = min(max(distance_r_hip / 100, 0.5), 0.5)
    distance_l_hip = calculate_distance(LHip, LAnkle)
    font_scale_l_hip = min(max(distance_l_hip / 100, 0.5), 0.5)

    # Draw angles
    if is_valid_point(RShoulder) and is_valid_point(RElbow) and is_valid_point(RWrist):
        if r_arm_angle is not None:
            draw_interior_sector(img, tuple(map(int, RElbow)), tuple(map(int, RShoulder)), tuple(map(int, RWrist)), "right")
            draw_text_with_outline(img, f"{r_arm_angle:.1f}", tuple(map(int, RElbow)), font_scale_r, 1)

    if is_valid_point(LShoulder) and is_valid_point(LElbow) and is_valid_point(LWrist):
        if l_arm_angle is not None:
            draw_interior_sector(img, tuple(map(int, LElbow)), tuple(map(int, LShoulder)), tuple(map(int, LWrist)), "left")
            draw_text_with_outline(img, f"{l_arm_angle:.1f}", tuple(map(int, LElbow)), font_scale_l, 1)

    if is_valid_point(LShoulder) and is_valid_point(RShoulder) and is_valid_point(RElbow):
        if r_shoulder_angle is not None:
            draw_interior_sector(img, tuple(map(int, RShoulder)), tuple(map(int, LShoulder)), tuple(map(int, RElbow)), "right")
            draw_text_with_outline(img, f"{r_shoulder_angle:.1f}", tuple(map(int, RShoulder)), font_scale_r_shoulder, 1)

    if is_valid_point(RShoulder) and is_valid_point(LShoulder) and is_valid_point(LElbow):
        if l_shoulder_angle is not None:
            draw_interior_sector(img, tuple(map(int, LShoulder)), tuple(map(int, RShoulder)), tuple(map(int, LElbow)), "left")
            draw_text_with_outline(img, f"{l_shoulder_angle:.1f}", tuple(map(int, LShoulder)), font_scale_l_shoulder, 1)

    if is_valid_point(RShoulder) and is_valid_point(RHip) and is_valid_point(RAnkle):
        if r_hip_angle is not None:
            draw_interior_sector(img, tuple(map(int, RHip)), tuple(map(int, RShoulder)), tuple(map(int, RAnkle)), "right")
            draw_text_with_outline(img, f"{r_hip_angle:.1f}", tuple(map(int, RHip)), font_scale_r_hip, 1)

    if is_valid_point(LShoulder) and is_valid_point(LHip) and is_valid_point(LAnkle):
        if l_hip_angle is not None:
            draw_interior_sector(img, tuple(map(int, LHip)), tuple(map(int, LShoulder)), tuple(map(int, LAnkle)), "left")
            draw_text_with_outline(img, f"{l_hip_angle:.1f}", tuple(map(int, LHip)), font_scale_l_hip, 1)

    return img
    
    ## New function to calculate the angle for drawing the sector on the interior angle side, Tommy, 05-01-2024            
def calculate_clockwise_angle_from_x_axis(center, pt):
    
    if center[1] == pt[1]:  # Check for division by zero
        return 0 if pt[0] >= center[0] else 180  # Return 0 or 180 based on the x-coordinate
    if center[0] == pt[0]:  # Check for division by zero
        return 90 if pt[1] > center[1] else 270 # Return 90 or 270 based on the y-coordinate

    angle_radian: float = 0
    angle_deg: float = 0
        
    if (pt[0] >= center[0] and pt[1] > center[1]): # Quadrant I
        angle_radian = math.atan((pt[1] - center[1]) / (pt[0] - center[0])) # arctan(y/x)
        angle_deg = angle_radian * 180 / math.pi # Convert to degrees
    elif (pt[0] < center[0] and pt[1] >= center[1]): # Quadrant II
        angle_radian = math.atan((center[0] - pt[0]) / (pt[1] - center[1])) # arctan(x/y)
        angle_deg = 90 + angle_radian * 180 / math.pi # Convert to degrees
    elif (pt[0] <= center[0] and pt[1] < center[1]): # Quadrant III
        angle_radian = math.atan((center[1] - pt[1]) / (center[0] - pt[0])) # arctan(y/x)
        angle_deg = 180 + angle_radian * 180 / math.pi # Convert to degrees
    elif (pt[0] > center[0] and pt[1] <= center[1]): # Quadrant IV
        angle_radian = math.atan((pt[0] - center[0]) / (center[1] - pt[1])) # arctan(x/y)
        angle_deg = 270 + angle_radian * 180 / math.pi # Convert to degrees

    return angle_deg


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