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
    draw_angles,
)
import shutil
import psutil
import time
from collections import deque
import tkinter as tk
from tkinter import filedialog
import customtkinter

EXPORTS_DIR = "detection/exports"
os.makedirs(EXPORTS_DIR, exist_ok=True)

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

selected_video_path = None
file_path = None
base_name = None
pose_model = None
result_label = None

def open_file_dialog():
    root = tk.Tk()
    root.withdraw()
    selected_video_path = filedialog.askopenfilename(title="Choose a video file")
    root.destroy()
    return selected_video_path

def btn_import_video():
    print("button pressed: Import video...")
    global selected_video_path
    selected_video_path = open_file_dialog()
    print("selected video path:", selected_video_path)
    run_video_tracking()

def run_video_tracking():
    print("demo run_video_tracking...")
    process_video(selected_video_path)
    button_ready_for_realtime = customtkinter.CTkButton(
        app,
        text="Are you ready?",
        width=300,
        height=50,
        command=run_realtime_tracking,
    )
    button_ready_for_realtime.grid(row=2, column=0, padx=0, pady=(20, 10))

def calculate_distance_statistics(data):
    abs_distances = [
        item["abs_distance"]
        for sublist in data
        for item in sublist
        if isinstance(item, dict) and item.get("abs_distance") is not None and item["abs_distance"] != float("inf")
    ]
    if not abs_distances:
        return 0, 0, 0, 0, 0
    max_distance = np.max(abs_distances)
    min_distance = np.min(abs_distances)
    mean_distance = np.mean(abs_distances)
    median_distance = np.median(abs_distances)
    std_distance = np.std(abs_distances)
    return max_distance, min_distance, mean_distance, median_distance, std_distance

def calculate_rms_metrics(matching_kpts_report, base_name):
    """Calculate RMS metrics and matching quality from process_webcam's matching_kpts_report."""
    max_distance, min_distance, mean_distance, median_distance, std_distance = (
        calculate_distance_statistics([matching_kpts_report])
    )

    individual_rms = [
        kpt.get("abs_distance", 0)
        for kpt in matching_kpts_report
        if isinstance(kpt, dict) and kpt.get("abs_distance") is not None and kpt["abs_distance"] != float("inf")
    ]
    overall_rms = np.mean(individual_rms) if individual_rms else 0

    threshold = 80 / 1280  # Normalized equivalent of 80 pixels
    low_rms_count = sum(1 for rms in individual_rms if rms <= threshold)
    total_keypoints = len(individual_rms)
    percentage_low_rms = (
        (low_rms_count / total_keypoints) * 100 if total_keypoints > 0 else 0
    )

    if percentage_low_rms > 80:
        matching_quality = "Perfect matching"
    elif 50 <= percentage_low_rms <= 80:
        matching_quality = "Good matching"
    elif 30 <= percentage_low_rms < 50:
        matching_quality = "Not matching enough"
    else:
        matching_quality = "Poor matching"

    overall_rms_threshold = 40 / 1280  # Normalized equivalent of 40 pixels
    if overall_rms <= overall_rms_threshold:
        overall_quality = "Overall perfect"
    elif (60 / 1280) > overall_rms > overall_rms_threshold:
        overall_quality = "Overall not bad"
    else:
        overall_quality = "Overall poor"

    print(
        f"Webcam RMS Metrics - Percentage Low RMS: {percentage_low_rms:.2f}%, "
        f"Overall RMS: {overall_rms:.6f}, Matching Quality: {matching_quality}, "
        f"Overall Quality: {overall_quality}, Total Keypoints: {total_keypoints}"
    )

    return {
        "max_distance": max_distance,
        "min_distance": min_distance,
        "mean_distance": mean_distance,
        "median_distance": median_distance,
        "std_distance": std_distance,
        "overall_rms": overall_rms,
        "percentage_low_rms": percentage_low_rms,
        "matching_quality": matching_quality,
        "overall_quality": overall_quality,
    }

def run_realtime_tracking(camera_id=0):
    print("demo run_realtime_tracking...")
    global result_label
    if not run_pose_init(file_path, camera_id):
        print("Pose initialization failed. Aborting webcam processing.")
        return
    process_webcam(file_path, base_name, camera_id)

def run_pose_init(file_path, camera_id=0):
    print("demo run_pose_init...")
    global pose_model
    visibility_threshold = 0.8
    pose_model = mp_pose.Pose(
        min_detection_confidence=visibility_threshold,
        min_tracking_confidence=visibility_threshold,
        model_complexity=1,
        smooth_landmarks=True,
    )
    cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        print("Error: Cannot access webcam.")
        return False

    with open(f"{file_path}/tracking_frame_report.json", "r") as file:
        ref_list = json.load(file)
    if not ref_list:
        print("Error: Reference JSON is empty.")
        cap.release()
        return False
    init_keypoints = ref_list[0]["keypoints"]

    pose_init_duration = 20
    pixel_threshold = 80
    visibility_threshold = 0.8
    init_time = time.time()
    continue_init = True
    frame_id = 0
    fps_time = time.time()

    with mp_pose.Pose(
        min_detection_confidence=visibility_threshold,
        min_tracking_confidence=visibility_threshold,
    ) as pose:
        while continue_init:
            success, frame = cap.read()
            if not success:
                print("Ignoring empty camera frame.")
                continue

            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            results = pose.process(image)
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            resolution = [frame_width, frame_height]

            current_time = time.time()
            fps = round(1.0 / (current_time - fps_time), 2)
            fps_time = current_time

            webcam_keypoints = extract_keypoints(results, frame_id)
            img_with_skeleton = image.copy()

            all_keypoints_matched = False
            if results.pose_landmarks:
                mp_drawing.draw_landmarks(
                    img_with_skeleton,
                    results.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS,
                    mp_drawing.DrawingSpec(
                        color=(245, 117, 66), thickness=2, circle_radius=2
                    ),
                    mp_drawing.DrawingSpec(
                        color=(245, 66, 230), thickness=2, circle_radius=2
                    ),
                )

                bbox = results.pose_landmarks.landmark
                x_min = min([lm.x for lm in bbox]) * frame_width
                y_min = min([lm.y for lm in bbox]) * frame_height
                x_max = max([lm.x for lm in bbox]) * frame_width
                y_max = max([lm.y for lm in bbox]) * frame_height
                cv2.rectangle(
                    img_with_skeleton,
                    (int(x_min), int(y_min)),
                    (int(x_max), int(y_max)),
                    (0, 255, 0),
                    2,
                )

                img_with_skeleton = draw_angles(
                    img_with_skeleton, webcam_keypoints, frame_width, frame_height
                )
                img_with_skeleton = draw_skeleton(
                    img_with_skeleton, init_keypoints, webcam_keypoints
                )

                matched_count = 0
                total_keypoints = sum(
                    1
                    for kpt in init_keypoints
                    if kpt.get("visibility", 0) > visibility_threshold
                )
                distances = []
                for ref_kpt, web_kpt in zip(init_keypoints, webcam_keypoints):
                    if (
                        ref_kpt.get("visibility", 0) > visibility_threshold
                        and web_kpt.get("visibility", 0) > visibility_threshold
                    ):
                        ref_x, ref_y = (
                            ref_kpt["x"] * frame_width,
                            ref_kpt["y"] * frame_height,
                        )
                        web_x, web_y = (
                            web_kpt["x"] * frame_width,
                            web_kpt["y"] * frame_height,
                        )
                        abs_distance = (
                            (ref_x - web_x) ** 2 + (ref_y - web_y) ** 2
                        ) ** 0.5
                        distances.append(abs_distance)
                        if abs_distance <= pixel_threshold:
                            matched_count += 1
                    else:
                        distances.append(None)
                all_keypoints_matched = matched_count == total_keypoints
                print(
                    f"Init Frame {frame_id} - Matched keypoints: {matched_count}/{total_keypoints}, "
                    f"Distances: {[round(d, 2) if d is not None else 'N/A' for d in distances[:5]]}"
                )

            elapsed_time = current_time - init_time
            countdown = max(0, pose_init_duration - int(elapsed_time))
            cv2.putText(
                img_with_skeleton,
                f"Pose Init: {countdown}s",
                (30, 150),
                cv2.FONT_HERSHEY_SIMPLEX,
                2,
                (255, 255, 255),
                5,
            )

            cv2.imshow("Pose Init", img_with_skeleton)

            key = cv2.waitKey(1)
            if key == 27 or all_keypoints_matched:
                cv2.destroyAllWindows()
                cap.release()
                return True

            if elapsed_time >= pose_init_duration and not all_keypoints_matched:
                cv2.destroyAllWindows()

                def create_modal():
                    modal = customtkinter.CTkToplevel()
                    modal.geometry("400x200")
                    modal.title("Pose Initialization Timeout")
                    modal.attributes("-topmost", True)

                    label = customtkinter.CTkLabel(
                        modal,
                        text="Do you want to try again or start the exercise?",
                        font=("Arial", 14),
                    )
                    label.pack(pady=20)

                    def continue_pose_init():
                        nonlocal continue_init, init_time
                        continue_init = True
                        init_time = time.time()
                        modal.destroy()

                    def start_tracking():
                        nonlocal continue_init
                        continue_init = False
                        modal.destroy()

                    btn_continue = customtkinter.CTkButton(
                        modal,
                        text="Retry",
                        width=150,
                        command=continue_pose_init,
                    )
                    btn_continue.pack(side="left", padx=20, pady=10)

                    btn_start = customtkinter.CTkButton(
                        modal,
                        text="Start Exercise",
                        width=150,
                        command=start_tracking,
                    )
                    btn_start.pack(side="right", padx=20, pady=10)

                    modal.grab_set()
                    modal.wait_window()

                create_modal()
                if continue_init:
                    cv2.imshow("Pose Init", img_with_skeleton)
                else:
                    cv2.destroyAllWindows()
                    cap.release()
                    return True

            frame_id += 1

        cap.release()
        return False

def generate_filename(base_name):
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

    os.path.join(export_path, "tracking_frame_report.json")
    os.path.join(export_path, "summary_report.json")

    return export_path

def process_webcam(file_path=None, base_name=None, camera_id=1):
    global result_label
    cap = cv2.VideoCapture(camera_id)
    tracking_frame_report = []
    matching_kpts_report = []
    match_percentages = []
    frame_id = 0
    fps_time = 0

    ref_list = []
    ref_keypoints = []
    visibility_threshold = 0.8

    start_time = time.time()
    with open(f"{file_path}/tracking_frame_report.json", "r") as file:
        ref_list = json.load(file)

    print("Start webcam processing...")

    with mp_pose.Pose(
        min_detection_confidence=visibility_threshold, min_tracking_confidence=visibility_threshold
    ) as pose:
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                print("Ignoring empty camera frame.")
                continue

            if frame_id > ref_list[-1]["frame_id"]:
                print("End of reference video.")
                break

            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            results = pose.process(image)
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            resolution = [frame_width, frame_height]

            current_time = time.time()
            fps = round(1.0 / (current_time - fps_time), 2)
            current_cpu_load = psutil.cpu_percent()

            webcam_keypoints = extract_keypoints(results, frame_id)
            tracking_frame_report.append(
                {
                    "filename": "webcam",
                    "frame_id": frame_id,
                    "fps": fps,
                    "cpu_load": current_cpu_load,
                    "resolution": resolution,
                    "keypoints": webcam_keypoints,
                    "frame_time": current_time,
                }
            )

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

                img_with_skeleton = image.copy()

                bbox = results.pose_landmarks.landmark
                x_min = min([lm.x for lm in bbox]) * frame_width
                y_min = min([lm.y for lm in bbox]) * frame_height
                x_max = max([lm.x for lm in bbox]) * frame_width
                y_max = max([lm.y for lm in bbox]) * frame_height
                cv2.rectangle(
                    img_with_skeleton,
                    (int(x_min), int(y_min)),
                    (int(x_max), int(y_max)),
                    (0, 255, 0),
                    2,
                )

                img_with_skeleton = draw_angles(
                    img_with_skeleton, webcam_keypoints, frame_width, frame_height
                )

                if frame_id < len(ref_list):
                    ref_keypoints = ref_list[frame_id]["keypoints"]
                else:
                    ref_keypoints = ref_list[frame_id % len(ref_list)]["keypoints"]

                matched_count = 0
                total_keypoints = sum(1 for kpt in ref_keypoints if kpt.get("visibility", 0) > visibility_threshold)
                frame_matching_kpts = []

                if ref_keypoints:
                    img_with_skeleton = draw_skeleton(
                        img_with_skeleton, ref_keypoints, webcam_keypoints
                    )
                    for kpt in ref_keypoints:
                        if "abs_distance" in kpt and kpt["abs_distance"] is not None and kpt["abs_distance"] != float("inf"):
                            frame_matching_kpts.append(kpt)
                            if kpt["abs_distance"] <= 80 / frame_width:
                                matched_count += 1

                    matching_kpts_report.extend(frame_matching_kpts)

                match_percentage = (matched_count / total_keypoints * 100) if total_keypoints > 0 else 0
                match_percentages.append(match_percentage)
                print(
                    f"Webcam Frame {frame_id} - Matched keypoints: {matched_count}/{total_keypoints}, "
                    f"Match Percentage: {match_percentage:.2f}%"
                )
                
                overall_score = int(np.round(np.mean(match_percentages))) if match_percentages else 0
                console_log(
                    img_with_skeleton,
                    {
                        "filename": "webcam",
                        "frame_id": frame_id,
                        "resolution": resolution,
                        "frame_time": current_time,
                        "fps": fps,
                        "cpu_load": current_cpu_load,
                        "ref_video": base_name or ref_list[0]["filename"],
                        "match_percentage": f"{match_percentage:.2f}%",
                    },
                    additional_text=f"Overall Score: {overall_score}"
                )

                cv2.imshow("MediaPipe Pose - Webcam", img_with_skeleton)
                if cv2.waitKey(5) & 0xFF == 27:
                    break

            frame_id += 1
            fps_time = current_time

    total_time = time.time() - start_time
    overall_score = np.mean(match_percentages) if match_percentages else 0
    print(f"Total webcam processing time: {total_time:.2f} seconds")
    print(f"Overall Score: {overall_score}")

    rms_metrics = calculate_rms_metrics(matching_kpts_report, base_name)

    if result_label is None:
        result_label = customtkinter.CTkLabel(
            app,
            text="",
            font=("Arial", 14),
            fg_color="transparent",
        )
        result_label.grid(row=3, column=0, padx=0, pady=(10, 20))
    result_label.configure(
        text=f"Matching Quality: {rms_metrics['matching_quality']}\n"
             f"Overall RMS Quality: {rms_metrics['overall_quality']}\n"
             f"Overall Score: {int(overall_score)}"
    )

    webcam_file_path = generate_filename(
        "webcam_" + (base_name or ref_list[0]["filename"])
    )
    save_keypoints_to_json(
        tracking_frame_report, f"{webcam_file_path}/tracking_frame_report.json"
    )
    save_keypoints_to_json(
        matching_kpts_report, f"{webcam_file_path}/matching_kpts_report.json"
    )

    avg_fps = (
        sum([frame["fps"] for frame in tracking_frame_report])
        / len(tracking_frame_report)
        if tracking_frame_report
        else 0
    )
    avg_cpu_load = (
        sum([frame["cpu_load"] for frame in tracking_frame_report])
        / len(tracking_frame_report)
        if tracking_frame_report
        else 0
    )

    summary_report = {
        "datetime": datetime.datetime.now().strftime("%d-%m-%Y_%H-%M-%S"),
        "filename": "webcam",
        "total_frames": frame_id,
        "total_time": total_time,
        "avg_fps": avg_fps,
        "avg_cpu_load": avg_cpu_load,
        "ref_video": base_name or ref_list[0]["filename"],
        "overall_score": overall_score,
        **rms_metrics,
    }

    save_summart_report_to_json(
        summary_report, f"{webcam_file_path}/summary_report.json"
    )

    cap.release()
    cv2.destroyAllWindows()
    return tracking_frame_report, matching_kpts_report

def process_video(video_path):
    global file_path, base_name
    cap = cv2.VideoCapture(video_path)
    keypoints_list = []
    base_name = os.path.splitext(os.path.basename(video_path))[0]
    frame_id = 0
    fps_time = 0
    tracking_frame_report = []
    summary_report = []
    visibility_threshold = 0.8

    start_time = time.time()
    print("Start processing video...")

    with mp_pose.Pose(
        min_detection_confidence=visibility_threshold, min_tracking_confidence=visibility_threshold
    ) as pose:
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                print("End of video.")
                break

            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            results = pose.process(image)
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            resolution = [frame_width, frame_height]

            current_time = time.time()
            fps = round(1.0 / (current_time - fps_time), 2)
            current_cpu_load = psutil.cpu_percent()

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
            )

            if results.pose_landmarks:
                mp_drawing.draw_landmarks(
                    image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS
                )

                console_log(
                    image,
                    {
                        "filename": base_name,
                        "frame_id": frame_id,
                        "resolution": resolution,
                        "frame_time": current_time,
                        "fps": fps,
                        "cpu_load": current_cpu_load,
                    },
                )

                bbox = results.pose_landmarks.landmark
                x_min = min([lm.x for lm in bbox]) * frame_width
                y_min = min([lm.y for lm in bbox]) * frame_height
                x_max = max([lm.x for lm in bbox]) * frame_width
                y_max = max([lm.y for lm in bbox]) * frame_height
                cv2.rectangle(
                    image,
                    (int(x_min), int(y_min)),
                    (int(x_max), int(y_max)),
                    (0, 255, 0),
                    2,
                )

                image = draw_angles(image, keypoints, frame_width, frame_height)

            cv2.imshow("MediaPipe Pose - Video", image)
            if cv2.waitKey(5) & 0xFF == 27:
                break

            frame_id += 1
            fps_time = time.time()

    total_time = time.time() - start_time
    print(f"Total video processing time: {total_time:.2f} seconds")

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
    image = cv2.imread(image_path)
    visibility_threshold = 0.8
    with mp_pose.Pose(
        min_detection_confidence=visibility_threshold, min_tracking_confidence=visibility_threshold
    ) as pose:
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = pose.process(image_rgb)

        if results.pose_landmarks:
            mp_drawing.draw_landmarks(
                image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS
            )

        cv2.imshow("MediaPipe Pose - Image", image)
        cv2.waitKey(0)

    cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MediaPipe Pose Estimation Demo")
    parser.add_argument(
        "--video",
        type=str,
        default="",
        help="Path to video file or camera id (0 for webcam)",
    )
    parser.add_argument(
        "--demo",
        type=int,
        default=0,
        help="1: demo mode with UI. (default: 0)",
    )
    parser.add_argument("--image", type=str, default="", help="Path to image file")
    parser.add_argument(
        "--camera-id",
        type=int,
        default=0,
        help="Webcam ID for sliding window (default: 0)",
    )
    parser.add_argument(
        "--reference-video-json",
        type=str,
        default="detection/temp_report/tracking_frame_report_video-1.json",
        help="reference video JSON tracking report for realtime tracking",
    )

    args = parser.parse_args()

    if args.demo == 1:
        print("demo mode is on")
        customtkinter.set_appearance_mode("light")
        app = customtkinter.CTk()
        app.geometry("1080x607")
        app.title("MediaPipe Demo")

        label = customtkinter.CTkLabel(
            app,
            text="Choose your exercise video for tracking",
            fg_color="transparent",
            font=("Arial", 20),
        )
        button = customtkinter.CTkButton(
            app, text="Import", width=200, command=btn_import_video
        )

        app.grid_columnconfigure(0, weight=1)
        label.grid(row=0, column=0, padx=0, pady=(20, 10))
        button.grid(row=1, column=0, padx=0, pady=0)

        app.mainloop()

    else:
        if args.video:
            process_video(args.video)
        elif args.image:
            process_image(args.image)
        else:
            process_webcam(args.camera_id)