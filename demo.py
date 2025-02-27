import cv2
import os
import mediapipe as mp
import numpy as np
import json
import argparse
import datetime
from pose import (
    extract_keypoints,
    save_keypoints_to_json,
    process_pose,
    save_summart_report_to_json,
    console_log,
    draw_skeleton,
)  # Import your new functions
import shutil
import psutil
import time

EXPORTS_DIR = "detection/exports"
os.makedirs(EXPORTS_DIR, exist_ok=True)

# Initialize MediaPipe Pose
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose


def generate_filename(base_name):
    """Generate a unique filename based on the base name and current timestamp."""
    # timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    # return os.path.join(EXPORTS_DIR, f"{base_name}_{timestamp}.json")
    current_datetime = datetime.datetime.now().strftime("%Y%m%d_%H-%M-%S")
    try:
        filename = os.path.basename(base_name)
    except:
        filename = "webcam"
    export_path = f"{EXPORTS_DIR}/{filename}_{current_datetime}/"

    if not os.path.exists(f"{export_path}"):
        os.makedirs(f"{export_path}")
    else:
        shutil.rmtree(f"{export_path}")
        os.makedirs(f"{export_path}")

    ## Create a JSON file to store the motion-tracking keypoints list
    os.path.join(export_path, "tracking_frame_report.json")

    ## Create a json file to store the summary report
    os.path.join(export_path, "summary_report.json")

    return export_path


def process_webcam():
    """Process webcam video feed."""
    # Start capturing video
    cap = cv2.VideoCapture(0)
    tracking_frame_report = []
    summary_report = []
    frame_id = 0
    fps_time = 0
    
    ref_list = []
    ref_keypoints = []

    start_time = time.time()
    ## Load the JSON data from the file
    with open("detection/tracking_frame_report_video-2.json", "r") as file:
        ref_list = json.load(file)
    print("Start processing...")

    with mp_pose.Pose(
        min_detection_confidence=0.5, min_tracking_confidence=0.5
    ) as pose:
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

            ## Get the resolution of the captured image
            frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            resolution = [frame_width, frame_height]

            ## Calculate the fps
            current_time = time.time()
            fps = round(1.0 / (current_time - fps_time), 2)

            ## Access the CPU usage
            current_cpu_load = psutil.cpu_percent()

            # # Extract landmarks
            # try:
            #     landmarks = results.pose_landmarks.landmark
            #     # print(landmarks)
            # except:
            #     pass

            # Extract landmarks
            keypoints = extract_keypoints(results, frame_id)
            tracking_frame_report.append(
                {
                    "filename": "webcam",
                    "frame_id": frame_id,
                    "fps": fps,
                    "cpu_load": current_cpu_load,
                    "resolution": resolution,
                    "keypoints": keypoints,
                }
            )  # Store the object

            # Render detections
            if results.pose_landmarks:
                mp_drawing.draw_landmarks(
                    image,
                    results.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS,
                    mp_drawing.DrawingSpec(
                        color=(245, 117, 66), thickness=2, circle_radius=2
                    ),
                    mp_drawing.DrawingSpec(
                        color=(245, 66, 230), thickness=2, circle_radius=2
                    ),
                )
                
                ## Draw the skeleton
                if frame_id < len(ref_list):
                    ref_keypoints = ref_list[frame_id]["keypoints"]
                else:
                    ref_keypoints = ref_list[frame_id % len(ref_list)]["keypoints"]
                    
                img_with_skeleton = draw_skeleton(image, ref_keypoints)
                
                console_log(
                    img_with_skeleton,
                    {
                        "filename": "webcam",
                        "frame_id": frame_id,
                        "resolution": resolution,
                        "frame_time": current_time,
                        "fps": fps,
                        "cpu_load": current_cpu_load,
                    },
                )

            cv2.imshow("MediaPipe Pose - Webcam", img_with_skeleton)
            if cv2.waitKey(5) & 0xFF == 27:  # Press 'Esc' to exit
                break

            frame_id += 1
            fps_time = time.time()

    total_time = time.time() - start_time
    print(f"Total processing time: {total_time:.2f} seconds")

    # Save keypoints to JSON file before exiting
    file_path = generate_filename("webcam")
    save_keypoints_to_json(
        tracking_frame_report, f"{file_path}/tracking_frame_report.json"
    )

    avg_fps = sum([frame["fps"] for frame in tracking_frame_report]) / len(
        tracking_frame_report
    )
    avg_cpu_load = sum([frame["cpu_load"] for frame in tracking_frame_report]) / len(
        tracking_frame_report
    )

    summary_report = {
        "total_frames": frame_id,
        "total_time": total_time,
        "avg_fps": avg_fps,
        "avg_cpu_load": avg_cpu_load,
    }

    save_summart_report_to_json(summary_report, f"{file_path}/summary_report.json")

    cap.release()
    cv2.destroyAllWindows()


def process_video(video_path):
    """Process video file."""
    cap = cv2.VideoCapture(video_path)
    keypoints_list = []
    base_name = os.path.splitext(os.path.basename(video_path))[
        0
    ]  # Get the base name of the video file
    frame_id = 0
    fps_time = 0
    tracking_frame_report = []
    summary_report = []

    start_time = time.time()
    print("Start processing...")

    with mp_pose.Pose(
        min_detection_confidence=0.5, min_tracking_confidence=0.5
    ) as pose:
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

            ## Get the resolution of the captured image
            frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            resolution = [frame_width, frame_height]

            ## Calculate the fps
            current_time = time.time()
            fps = round(1.0 / (current_time - fps_time), 2)

            ## Access the CPU usage
            current_cpu_load = psutil.cpu_percent()

            # Extract landmarks
            keypoints = extract_keypoints(results, frame_id)
            tracking_frame_report.append(
                {
                    "filename": base_name,
                    "frame_id": frame_id,
                    "fps": fps,
                    "cpu_load": current_cpu_load,
                    "resolution": resolution,
                    "keypoints": keypoints,
                }
            )  # Store the object

            # Draw landmarks
            if results.pose_landmarks:
                mp_drawing.draw_landmarks(
                    image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS
                )

                console_log(
                    image,
                    {
                        "filename": "webcam",
                        "frame_id": frame_id,
                        "resolution": resolution,
                        "frame_time": current_time,
                        "fps": fps,
                        "cpu_load": current_cpu_load,
                    },
                )

            cv2.imshow("MediaPipe Pose - Video", image)
            if cv2.waitKey(5) & 0xFF == 27:  # Press 'Esc' to exit
                break

            frame_id += 1
            fps_time = time.time()

    total_time = time.time() - start_time
    print(f"Total processing time: {total_time:.2f} seconds")

    # Save keypoints to JSON file before exiting
    file_path = generate_filename(base_name)
    save_keypoints_to_json(
        tracking_frame_report, f"{file_path}/tracking_frame_report.json"
    )

    avg_fps = sum([frame["fps"] for frame in tracking_frame_report]) / len(
        tracking_frame_report
    )
    avg_cpu_load = sum([frame["cpu_load"] for frame in tracking_frame_report]) / len(
        tracking_frame_report
    )

    summary_report = {
        "total_frames": frame_id,
        "total_time": total_time,
        "avg_fps": avg_fps,
        "avg_cpu_load": avg_cpu_load,
    }

    save_summart_report_to_json(summary_report, f"{file_path}/summary_report.json")

    cap.release()
    cv2.destroyAllWindows()


def process_image(image_path):
    """Process a single image."""
    image = cv2.imread(image_path)
    with mp_pose.Pose(
        min_detection_confidence=0.5, min_tracking_confidence=0.5
    ) as pose:
        # Convert the BGR image to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = pose.process(image_rgb)

        # Draw landmarks
        if results.pose_landmarks:
            mp_drawing.draw_landmarks(
                image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS
            )

        cv2.imshow("MediaPipe Pose - Image", image)
        cv2.waitKey(0)  # Press any key to close the window

    cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MediaPipe Pose Estimation Demo")
    parser.add_argument(
        "--video",
        type=str,
        default="",
        help="Path to video file or camera id (0 for webcam)",
    )
    parser.add_argument("--image", type=str, default="", help="Path to image file")

    args = parser.parse_args()

    if args.video:
        process_video(args.video)
    elif args.image:
        process_image(args.image)
    else:
        process_webcam()
