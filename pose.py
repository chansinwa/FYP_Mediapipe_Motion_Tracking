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

BODY_PARTS_CONNECTIONS = [
    [0, 1],
    [0, 4],
    [1, 2],
    [2, 3],
    [4, 5],
    [5, 6],
    [3, 7],
    [6, 8],
    [9, 10],
    [11, 12],
    [11, 13],
    [13, 15],
    [12, 14],
    [14, 16],
    [11, 23],
    [12, 24],
    [23, 24],
    [23, 25],
    [25, 27],
    [24, 26],
    [26, 28],
    [15, 17],
    [15, 19],
    [15, 21],
    [16, 18],
    [16, 20],
    [16, 22],
    [27, 29],
    [29, 31],
    [28, 30],
    [30, 32],
]

def extract_keypoints(results, frame_id):
    """Extract keypoints from MediaPipe results."""
    keypoints = []
    if results.pose_landmarks:
        for index, landmark in enumerate(results.pose_landmarks.landmark):
            keypoints.append(
                {
                    "frame_id": frame_id,
                    "kpt_id": index,
                    "name": PoseLandmark(index).name,
                    "x": landmark.x,
                    "y": landmark.y,
                    "z": landmark.z,
                    "visibility": landmark.visibility,
                }
            )
    return keypoints

def save_keypoints_to_json(keypoints, filename):
    """Save keypoints to a JSON file."""
    with open(filename, "w") as f:
        json.dump(keypoints, f, indent=4)

def save_summart_report_to_json(report, filename):
    """Save the summary report to a JSON file."""
    with open(filename, "w") as f:
        json.dump(report, f, indent=4)

def process_pose(image):
    """Process the pose detection and return keypoints."""
    with mp_pose.Pose(
        min_detection_confidence=0.5, min_tracking_confidence=0.5
    ) as pose:
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_rgb.flags.writeable = False
        results = pose.process(image_rgb)
        image_rgb.flags.writeable = True
        return results

def console_log(img, msg, additional_text=None):
    """Render console messages on the image with optional additional text below."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    org = (10, 30)
    font_color = (0, 255, 0)
    line_type = 2
    line_spacing = 20

    y = org[1]
    for key, value in msg.items():
        line = f"{key}: {value}"
        cv2.putText(img, line, (org[0], y), font, font_scale, font_color, line_type)
        y += line_spacing

    if additional_text:
        cv2.putText(
            img,
            additional_text,
            (org[0], y + line_spacing),
            font,
            font_scale,
            font_color,
            line_type
        )

    return img

def calculate_distance(point1, point2):
    """Calculate the Euclidean distance between two points."""
    return np.linalg.norm(np.array(point1) - np.array(point2))

def draw_skeleton(img, ref_frame_kpts, webcam_kpts_list=None):
    """Draw a semi-transparent grey skeleton based on keypoints."""
    overlay = img.copy()
    line_color = (128, 128, 128)
    line_thickness = 60
    height, width = img.shape[:2]
    keypoints_dict = {
        kpt["kpt_id"]: (int(kpt["x"] * width), int(kpt["y"] * height))
        for kpt in ref_frame_kpts
        if kpt["visibility"] > 0.5
    }

    for connection in BODY_PARTS_CONNECTIONS:
        idx1, idx2 = connection
        if idx1 in keypoints_dict and idx2 in keypoints_dict:
            pt1 = keypoints_dict[idx1]
            pt2 = keypoints_dict[idx2]
            cv2.line(overlay, pt1, pt2, line_color, line_thickness)

    alpha = 0.5
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)

    if webcam_kpts_list:
        circle_coordinates = []
        circle_colors = []

        for ref_kpt in ref_frame_kpts:
            if ref_kpt["visibility"] > 0.5:
                kpt_id = ref_kpt["kpt_id"]
                ref_x, ref_y = int(ref_kpt["x"] * width), int(ref_kpt["y"] * height)
                ref_abs_coords = (ref_x, ref_y)
                ref_kpt["ref_abs_coords"] = ref_abs_coords

                webcam_kpt = next(
                    (kpt for kpt in webcam_kpts_list if kpt["kpt_id"] == kpt_id), None
                )
                if webcam_kpt and webcam_kpt["visibility"] > 0.5:
                    webcam_x, webcam_y = int(webcam_kpt["x"] * width), int(
                        webcam_kpt["y"] * height
                    )
                    webcam_abs_coords = (webcam_x, webcam_y)

                    pixel_distance = calculate_distance(ref_abs_coords, webcam_abs_coords)
                    normalized_distance = pixel_distance / width
                    ref_kpt["webcam_coords"] = webcam_abs_coords
                    ref_kpt["abs_distance"] = normalized_distance

                    if pixel_distance <= 80:
                        circle_coordinates.append(ref_abs_coords)
                        circle_colors.append((0, 255, 0))
                    else:
                        circle_coordinates.append(ref_abs_coords)
                        circle_colors.append((0, 0, 255))
                else:
                    circle_coordinates.append(ref_abs_coords)
                    circle_colors.append((0, 0, 255))
                    ref_kpt["webcam_abs_coords"] = None
                    ref_kpt["abs_distance"] = float("inf")

        for coord, color in zip(circle_coordinates, circle_colors):
            cv2.circle(img, coord, 15, color, -1)

        color_mapping = {
            (128, 128, 128): "Reference motion",
            (0, 255, 0): "Correct joint posture",
            (0, 0, 255): "Incorrect joint posture",
        }
        indicator_size = 30
        text_offset = 10
        x_start = 20
        y_start = img.shape[0] - 120

        cv2.rectangle(
            img,
            (10, y_start - 10),
            (x_start + 260, img.shape[0]),
            (255, 255, 255),
            -1,
        )

        for idx, (color, label) in enumerate(color_mapping.items()):
            cv2.rectangle(
                img,
                (x_start, y_start + idx * (indicator_size + text_offset)),
                (
                    x_start + indicator_size,
                    y_start + indicator_size + idx * (indicator_size + text_offset),
                ),
                color,
                -1,
            )
            cv2.putText(
                img,
                label,
                (
                    x_start + indicator_size + 10,
                    y_start + indicator_size + idx * (indicator_size + text_offset) - 5,
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 0),
                1,
                cv2.LINE_AA,
            )
    return img

def calculate_angle(a, b, c):
    """Calculate the angle in degrees at point b formed by points a-b-c."""
    if a is None or b is None or c is None:
        return None
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    a_length = np.linalg.norm(b - c)
    b_length = np.linalg.norm(a - c)
    c_length = np.linalg.norm(a - b)
    try:
        angle = np.arccos(
            (a_length**2 + c_length**2 - b_length**2) / (2 * a_length * c_length)
        )
        return np.degrees(angle)
    except (ValueError, ZeroDivisionError):
        return None

def is_valid_point(point):
    """Check if a point is valid (not None and has valid coordinates)."""
    return point is not None

def draw_text_with_outline(img, text, position, font_scale, thickness):
    """Draw text with a black outline and white fill."""
    cv2.putText(
        img,
        text,
        (position[0] - 1, position[1] - 1),
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        (0, 0, 0),
        thickness + 1,
    )
    cv2.putText(
        img,
        text,
        (position[0] + 1, position[1] - 1),
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        (0, 0, 0),
        thickness + 1,
    )
    cv2.putText(
        img,
        text,
        (position[0] - 1, position[1] + 1),
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        (0, 0, 0),
        thickness + 1,
    )
    cv2.putText(
        img,
        text,
        (position[0] + 1, position[1] + 1),
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        (0, 0, 0),
        thickness + 1,
    )
    cv2.putText(
        img,
        text,
        position,
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        (255, 255, 255),
        thickness,
    )

def draw_interior_sector(img, center, pt_a, pt_b, side=None):
    """Draw a filled sector for the interior angle."""
    radius = 30
    arc_color = (0, 127, 255)
    sector_color = (0, 127, 255)

    def calculate_clockwise_angle_from_x_axis(center, pt):
        dx = pt[0] - center[0]
        dy = pt[1] - center[1]
        angle = np.degrees(np.arctan2(dy, dx))
        return angle if angle >= 0 else angle + 360

    angle_a = calculate_clockwise_angle_from_x_axis(center, pt_a)
    angle_b = calculate_clockwise_angle_from_x_axis(center, pt_b)

    if angle_b < angle_a:
        angle_a, angle_b = angle_b, angle_a
    if angle_b - angle_a < 180:
        start_angle = angle_a
        end_angle = angle_b - angle_a
    else:
        start_angle = angle_b
        end_angle = 360 - (angle_b - angle_a)

    transparency = 0.4
    original_img = img.copy()
    cv2.ellipse(
        img, center, (radius, radius), start_angle, 0, end_angle, sector_color, -1
    )
    cv2.addWeighted(img, transparency, original_img, 1 - transparency, 0, img)
    cv2.ellipse(img, center, (radius, radius), start_angle, 0, end_angle, arc_color, 2)

def draw_angles(img, keypoints, width, height):
    """Draw angles for key joints on the image."""
    kpts_dict = {kpt["kpt_id"]: (int(kpt["x"] * width), int(kpt["y"] * height))
                 for kpt in keypoints if kpt["visibility"] > 0.5}

    def get_point(kpt_id):
        return kpts_dict.get(kpt_id, None)

    RShoulder = get_point(PoseLandmark.RIGHT_SHOULDER)
    RElbow = get_point(PoseLandmark.RIGHT_ELBOW)
    RWrist = get_point(PoseLandmark.RIGHT_WRIST)
    LShoulder = get_point(PoseLandmark.LEFT_SHOULDER)
    LElbow = get_point(PoseLandmark.LEFT_ELBOW)
    LWrist = get_point(PoseLandmark.LEFT_WRIST)
    Neck = get_point(PoseLandmark.NOSE)
    RHip = get_point(PoseLandmark.RIGHT_HIP)
    RAnkle = get_point(PoseLandmark.RIGHT_ANKLE)
    LHip = get_point(PoseLandmark.LEFT_HIP)
    LAnkle = get_point(PoseLandmark.LEFT_ANKLE)
    LKnee = get_point(PoseLandmark.LEFT_KNEE)
    RKnee = get_point(PoseLandmark.RIGHT_KNEE)

    r_arm_angle = calculate_angle(RShoulder, RElbow, RWrist)
    l_arm_angle = calculate_angle(LShoulder, LElbow, LWrist)
    r_shoulder_angle = calculate_angle(LShoulder, RShoulder, RElbow)
    l_shoulder_angle = calculate_angle(RShoulder, LShoulder, LElbow)
    r_hip_angle = calculate_angle(RShoulder, RHip, RKnee)
    l_hip_angle = calculate_angle(LShoulder, LHip, LKnee)
    l_knee_angle = calculate_angle(LHip, LKnee, LAnkle)
    r_knee_angle = calculate_angle(RHip, RKnee, RAnkle)

    font_scale_r_arm = min(max(calculate_distance(RElbow, RWrist) / 100, 0.5), 0.5) if RElbow and RWrist else 0.5
    font_scale_l_arm = min(max(calculate_distance(LElbow, LWrist) / 100, 0.5), 0.5) if LElbow and LWrist else 0.5
    font_scale_r_shoulder = min(max(calculate_distance(RShoulder, RElbow) / 100, 0.5), 0.5) if RShoulder and RElbow else 0.5
    font_scale_l_shoulder = min(max(calculate_distance(LShoulder, LElbow) / 100, 0.5), 0.5) if LShoulder and LElbow else 0.5
    font_scale_r_hip = min(max(calculate_distance(RHip, RAnkle) / 100, 0.5), 0.5) if RHip and RAnkle else 0.5
    font_scale_l_hip = min(max(calculate_distance(LHip, LAnkle) / 100, 0.5), 0.5) if LHip and LAnkle else 0.5

    if is_valid_point(RShoulder) and is_valid_point(RElbow) and is_valid_point(RWrist) and r_arm_angle:
        draw_interior_sector(img, RElbow, RShoulder, RWrist, "right")
        draw_text_with_outline(img, f"{r_arm_angle:.1f}", RElbow, font_scale_r_arm, 1)

    if is_valid_point(LShoulder) and is_valid_point(LElbow) and is_valid_point(LWrist) and l_arm_angle:
        draw_interior_sector(img, LElbow, LShoulder, LWrist, "left")
        draw_text_with_outline(img, f"{l_arm_angle:.1f}", LElbow, font_scale_l_arm, 1)

    if is_valid_point(LShoulder) and is_valid_point(RShoulder) and is_valid_point(RElbow) and r_shoulder_angle:
        draw_interior_sector(img, RShoulder, LShoulder, RElbow, "right")
        draw_text_with_outline(img, f"{r_shoulder_angle:.1f}", RShoulder, font_scale_r_shoulder, 1)

    if is_valid_point(RShoulder) and is_valid_point(LShoulder) and is_valid_point(LElbow) and l_shoulder_angle:
        draw_interior_sector(img, LShoulder, RShoulder, LElbow, "left")
        draw_text_with_outline(img, f"{l_shoulder_angle:.1f}", LShoulder, font_scale_l_shoulder, 1)

    if is_valid_point(RKnee) and is_valid_point(RHip) and is_valid_point(RAnkle) and r_knee_angle:
        draw_interior_sector(img, RKnee, RHip, RAnkle, "right")
        draw_text_with_outline(img, f"{r_knee_angle:.1f}", RKnee, font_scale_r_hip, 1)

    if is_valid_point(LKnee) and is_valid_point(LHip) and is_valid_point(LAnkle) and l_knee_angle:
        draw_interior_sector(img, LKnee, LHip, LAnkle, "left")
        draw_text_with_outline(img, f"{l_knee_angle:.1f}", LKnee, font_scale_l_hip, 1)
        
    if is_valid_point(RHip) and is_valid_point(RKnee) and is_valid_point(RShoulder) and r_hip_angle:
        draw_interior_sector(img, RHip, RShoulder, RKnee, "right")
        draw_text_with_outline(img, f"{r_hip_angle:.1f}", RHip, font_scale_r_hip, 1)
        
    if is_valid_point(LHip) and is_valid_point(LKnee) and is_valid_point(LShoulder) and l_hip_angle:
        draw_interior_sector(img, LHip, LShoulder, LKnee, "left")
        draw_text_with_outline(img, f"{l_hip_angle:.1f}", LHip, font_scale_l_hip, 1)

    return img