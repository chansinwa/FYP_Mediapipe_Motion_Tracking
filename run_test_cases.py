import subprocess
import os

# Directory containing the videos
VIDEO_DIR = "detection/videos"

# List of video filenames to process
video_id_list = [1, 2, 3, 4, 5, 6, 12]
# 37, 38, 14, 15, 16, 27, 28, 30, 33, 34, 36 (run overnight)
video_files = [f"video-{i}.mov" for i in video_id_list]

command_sliding_window = [
    "python",
    "parallel_demo.py",
    "--sliding",
    "--camera-id",
    "1",
]

command_video_tracking = [
    "python",
    "demo.py",
    "--camera-id",
    "1",
    "--video",
]

command_realtime_tracking = [
    "python",
    "demo.py",
    "--camera-id",
    "1",
    "--reference-video-json",
]

# Specific window sizes to use
window_sizes = [10]

# Ensure the video directory exists
if not os.path.exists(VIDEO_DIR):
    print(f"Error: Directory '{VIDEO_DIR}' does not exist.")
    exit(1)

# Run each command sequentially
for video_file in video_files:
    video_path = os.path.join(VIDEO_DIR, video_file)
    # Check if the video file exists
    if not os.path.exists(video_path):
        print(f"Skipping '{video_path}': File not found.")
        continue

    # Define the command list for this video
    command_list = [
        # Command 1: Real-time tracking with JSON output
        # command_realtime_tracking + [
        #     "detection/exports/temp_report/tracking_frame_report_" + video_file.replace(".mov", ".json")
        # ],
        # Command 2: Sliding window with varying window sizes
        *[
            command_sliding_window + ["--video", video_path, "--window-size", str(size)]
            for size in window_sizes
        ],
    ]

    # Execute each command in the list
    for command in command_list:
        print(f"Running command: {' '.join(command)}")
        try:
            # Run the command and wait for it to complete
            result = subprocess.run(command, check=True, text=True, capture_output=True)
            print(
                f"Finished processing '{video_path}' with command: {' '.join(command)}"
            )
            print(f"Output: {result.stdout}")
            if result.stderr:
                print(f"Errors: {result.stderr}")
        except subprocess.CalledProcessError as e:
            print(
                f"Error running '{video_path}' with command: {' '.join(command)}': {e}"
            )
            print(f"Output: {e.output}")
        except Exception as e:
            print(
                f"Unexpected error running '{video_path}' with command: {' '.join(command)}': {e}"
            )

print("All video processing completed.")
