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
    save_summart_report_to_json,
    console_log,
    draw_skeleton,
)
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
    current_datetime = datetime.datetime.now().strftime("%Y%m%d_%H-%M-%S")
    try:
        filename = os.path.basename(base_name)
    except:
        filename = "webcam"
    export_path = f"{EXPORTS_DIR}/{filename}_{current_datetime}/"
    if not os.path.exists(export_path):
        os.makedirs(export_path)
    else:
        shutil.rmtree(export_path)
        os.makedirs(export_path)
    return export_path


def extract_video_window(video_path, window_size, start_frame=0):
    """Extract keypoints and frames for a window of video frames."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file '{video_path}'")
        return None, None

    keypoints_window = deque(maxlen=window_size)
    frame_window = deque(maxlen=window_size)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    # print(f"Extracting video window from frame {start_frame} to {start_frame + window_size}")

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
            image_rgb.flags.writeable = True
            frame = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
            keypoints = extract_keypoints(results, frame_id)
            keypoints_window.append(keypoints)
            frame_window.append(frame)

    cap.release()
    return keypoints_window, frame_window


def process_sliding_window_side_by_side(video_path, window_size=5, camera_id=1):
    """Process video and webcam side-by-side with a sliding window."""
    # Webcam setup
    webcam_cap = cv2.VideoCapture(camera_id)
    if not webcam_cap.isOpened():
        print(f"Error: Could not open webcam with ID {camera_id}")
        return

    # Video setup
    video_cap = cv2.VideoCapture(video_path)
    if not video_cap.isOpened():
        print(f"Error: Could not open video file '{video_path}'")
        webcam_cap.release()
        return

    # Initial video window
    keypoints_window, frame_window = extract_video_window(video_path, window_size)
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
    video_frame_id = window_size - 1  # Start at the last frame of the initial window
    file_path = generate_filename("sliding_window_" + (video_path.split("/")[-1])[:-4])
    start_time = time.time()
    print(f"Starting side-by-side sliding window comparison with video: {video_path}")

    with mp_pose.Pose(
        min_detection_confidence=0.5, min_tracking_confidence=0.5
    ) as pose:
        while webcam_cap.isOpened():
            # Read webcam frame
            success, webcam_frame = webcam_cap.read()
            if not success:
                print("Ignoring empty webcam frame.")
                continue

            # Process webcam frame
            webcam_rgb = cv2.cvtColor(webcam_frame, cv2.COLOR_BGR2RGB)
            webcam_rgb.flags.writeable = False
            webcam_results = pose.process(webcam_rgb)
            webcam_rgb.flags.writeable = True
            webcam_frame = cv2.cvtColor(webcam_rgb, cv2.COLOR_RGB2BGR)

            # Extract webcam keypoints
            webcam_keypoints = extract_keypoints(webcam_results, frame_id)

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

            # Get video frame and process it for landmarks
            if frame_id < len(keypoints_window):
                video_frame = frame_window[frame_id]
                video_keypoints = keypoints_window[frame_id]
                # Re-process the frame to get results since we need landmarks
                video_rgb = cv2.cvtColor(video_frame, cv2.COLOR_BGR2RGB)
                video_rgb.flags.writeable = False
                video_results = pose.process(video_rgb)
                video_rgb.flags.writeable = True
                video_frame = cv2.cvtColor(video_rgb, cv2.COLOR_RGB2BGR)
            else:
                # Slide the window: process the next video frame
                video_frame_id += 1
                success, video_frame = video_cap.read()
                if success:
                    video_rgb = cv2.cvtColor(video_frame, cv2.COLOR_BGR2RGB)
                    video_rgb.flags.writeable = False
                    video_results = pose.process(video_rgb)
                    video_rgb.flags.writeable = True
                    video_frame = cv2.cvtColor(video_rgb, cv2.COLOR_RGB2BGR)
                    video_keypoints = extract_keypoints(video_results, video_frame_id)
                    keypoints_window.append(video_keypoints)
                    frame_window.append(video_frame)
                else:
                    print("End of video reached.")
                    break

            # Visualize video frame (left side)
            if video_results.pose_landmarks:
                mp_drawing.draw_landmarks(
                    video_frame,
                    video_results.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS,
                    mp_drawing.DrawingSpec(
                        color=(245, 117, 66), thickness=2, circle_radius=2
                    ),
                    mp_drawing.DrawingSpec(
                        color=(245, 66, 230), thickness=2, circle_radius=2
                    ),
                )
            
            console_log(
                video_frame,
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

            # Visualize webcam frame (right side) with video skeleton overlay
            webcam_with_skeleton = webcam_frame.copy()
            if webcam_results.pose_landmarks:
                mp_drawing.draw_landmarks(
                    webcam_with_skeleton,
                    webcam_results.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS,
                    mp_drawing.DrawingSpec(
                        color=(245, 117, 66), thickness=2, circle_radius=2
                    ),
                    mp_drawing.DrawingSpec(
                        color=(245, 66, 230), thickness=2, circle_radius=2
                    ),
                )
                webcam_with_skeleton = draw_skeleton(
                    webcam_with_skeleton, video_keypoints
                )
            
            console_log(
                webcam_with_skeleton,
                {
                    "filename": "webcam_vs_video",
                    "frame_id": frame_id,
                    "webcam_frame_id": frame_id,
                    "resolution": resolution,
                    "frame_time": current_time,
                    "fps": fps,
                    "cpu_load": current_cpu_load,
                },
            )

            # Resize frames to match heights and stack horizontally
            video_frame_resized = cv2.resize(video_frame, (frame_width, frame_height))
            combined_frame = np.hstack((video_frame_resized, webcam_with_skeleton))

            cv2.imshow(
                "Exercise Tutorial (Left) vs User Tracking (Right)", combined_frame
            )
            # cv2.imwrite(f"{file_path}frame_{frame_id}.jpg", combined_frame)

            if cv2.waitKey(5) & 0xFF == 27:  # Press 'Esc' to exit
                break

            frame_id += 1
            fps_time = current_time

    total_time = time.time() - start_time
    print(f"Total processing time: {total_time:.2f} seconds")

    # Save results
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
        "video_path": video_path,
        "window_size": window_size,
    }
    save_summart_report_to_json(summary_report, f"{file_path}/summary_report.json")

    video_cap.release()
    webcam_cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MediaPipe Pose Estimation Demo")
    parser.add_argument("--video", type=str, default="", help="Path to video file")
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
        default=1,
        help="Webcam ID for sliding window (default: 1)",
    )

    args = parser.parse_args()

    if args.sliding and args.video:
        process_sliding_window_side_by_side(
            args.video, window_size=args.window_size, camera_id=args.camera_id
        )
    # [Insert original elif conditions for process_video, process_image, process_webcam if needed]
