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
from collections import deque

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


def extract_video_window(video_path, window_size, start_frame=0):
    """Extract keypoints for a window of video frames."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file '{video_path}'")
        return None

    keypoints_window = deque(maxlen=window_size)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    with mp_pose.Pose(
        min_detection_confidence=0.5, min_tracking_confidence=0.5
    ) as pose:
        for frame_id in range(start_frame, start_frame + window_size):
            success, frame = cap.read()
            if not success:
                break
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image_rgb.flags.writeable = False
            results = pose.process(image_rgb)
            keypoints = extract_keypoints(results, frame_id)
            keypoints_window.append(keypoints)

    cap.release()
    return keypoints_window


def process_sliding_window(video_path, window_size=20, camera_id=0):
    """Process webcam and video with a sliding window for keypoint comparison."""
    # Webcam setup
    webcam_cap = cv2.VideoCapture(camera_id)
    if not webcam_cap.isOpened():
        print(f"Error: Could not open webcam with ID {camera_id}")
        return

    # Video window setup
    video_cap = cv2.VideoCapture(video_path)
    if not video_cap.isOpened():
        print(f"Error: Could not open video file '{video_path}'")
        webcam_cap.release()
        return

    # Initial video window
    keypoints_window = extract_video_window(video_path, window_size)
    if not keypoints_window or len(keypoints_window) < window_size:
        print(
            f"Warning: Video has fewer than {window_size} frames or failed to process"
        )
        video_cap.release()
        webcam_cap.release()
        return

    tracking_frame_report = []
    frame_id = 0
    fps_time = 0
    video_frame_id = 0
    start_time = time.time()
    print(f"Starting sliding window comparison with video: {video_path}")

    with mp_pose.Pose(
        min_detection_confidence=0.5, min_tracking_confidence=0.5
    ) as pose:
        while webcam_cap.isOpened():
            success, webcam_frame = webcam_cap.read()
            if not success:
                print("Ignoring empty webcam frame.")
                continue

            # Process webcam frame
            image = cv2.cvtColor(webcam_frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            results = pose.process(image)
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            # Extract webcam keypoints
            webcam_keypoints = extract_keypoints(results, frame_id)

            # Get resolution and performance metrics
            frame_width = int(webcam_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            frame_height = int(webcam_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            resolution = [frame_width, frame_height]
            current_time = time.time()
            fps = round(1.0 / (current_time - fps_time), 2) if frame_id > 0 else 0
            current_cpu_load = psutil.cpu_percent()

            # Store webcam tracking data
            tracking_frame_report.append(
                {
                    "filename": "webcam",
                    "frame_id": frame_id,
                    "fps": fps,
                    "cpu_load": current_cpu_load,
                    "resolution": resolution,
                    "keypoints": webcam_keypoints,
                }
            )

            # Get video keypoints from window
            if frame_id < len(keypoints_window):
                ref_keypoints = keypoints_window[frame_id]
            else:
                # Slide the window: process the next video frame
                video_frame_id += 1
                success, video_frame = video_cap.read()
                if success:
                    video_rgb = cv2.cvtColor(video_frame, cv2.COLOR_BGR2RGB)
                    video_rgb.flags.writeable = False
                    video_results = pose.process(video_rgb)
                    video_rgb.flags.writeable = True
                    ref_keypoints = extract_keypoints(video_results, video_frame_id)
                    keypoints_window.append(ref_keypoints)
                else:
                    print("End of video reached.")
                    break

            # Visualize
            img_with_skeleton = image.copy()
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
                img_with_skeleton = draw_skeleton(image, ref_keypoints)

            console_log(
                img_with_skeleton,
                {
                    "filename": "webcam_vs_video",
                    "frame_id": frame_id,
                    "video_frame_id": video_frame_id,
                    "resolution": resolution,
                    "frame_time": current_time,
                    "fps": fps,
                    "cpu_load": current_cpu_load,
                },
            )

            cv2.imshow("Sliding Window Pose Comparison", img_with_skeleton)
            if cv2.waitKey(5) & 0xFF == 27:  # Press 'Esc' to exit
                break

            frame_id += 1
            fps_time = current_time

    total_time = time.time() - start_time
    print(f"Total processing time: {total_time:.2f} seconds")

    # Save results
    file_path = generate_filename("sliding_window")
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

    video_cap.release()
    webcam_cap.release()
    cv2.destroyAllWindows()


def process_webcam():
    """Process webcam video feed."""
    # Start webcam with id 0 or 1 or 2
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

            # Default to the original image
            img_with_skeleton = image.copy()
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
    parser.add_argument(
        "--sliding",
        action="store_true",
        help="Use sliding window comparison with video",
    )
    parser.add_argument(
        "--window-size",
        type=int,
        default=20,
        help="Size of the sliding window (default: 20)",
    )
    parser.add_argument(
        "--camera-id",
        type=int,
        default=0,
        help="Webcam ID for sliding window (default: 0)",
    )

    args = parser.parse_args()

    if args.sliding and args.video:
        process_sliding_window(
            args.video, window_size=args.window_size, camera_id=args.camera_id
        )
    elif args.video:
        process_video(args.video)
    elif args.image:
        process_image(args.image)
    else:
        process_webcam()
