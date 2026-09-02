import os
import glob
import cv2
from tqdm import tqdm

RAW_VIDEO_DIR = "raw_videos"
OUTPUT_BASE_DIR = "dataset_tasks"

# Scene prefixes mapped to sequential lifelong learning tasks
TASK_MAPPING = {
    "VIRAT_S_0000": "task_1_scene0000",
    "VIRAT_S_0002": "task_2_scene0002",
    "VIRAT_S_0100": "task_3_scene0100"
}

def get_task_folder(filename):
    for prefix, task_name in TASK_MAPPING.items():
        if filename.startswith(prefix):
            return task_name
    return "task_other"

def extract_frames_from_video(video_path, output_dir, target_fps=1):
    os.makedirs(output_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"Error opening video: {video_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30.0
        
    sample_interval = max(1, int(round(fps / target_fps)))
    frame_idx = 0
    saved_count = 0

    while True:
        success, frame = cap.read()
        if not success:
            break
        if frame_idx % sample_interval == 0:
            frame_filename = os.path.join(output_dir, f"frame_{saved_count:05d}.jpg")
            cv2.imwrite(frame_filename, frame)
            saved_count += 1
        frame_idx += 1

    cap.release()

def process_all_videos():
    video_files = glob.glob(os.path.join(RAW_VIDEO_DIR, "*.mp4"))
    if not video_files:
        print(f"No .mp4 files found in '{RAW_VIDEO_DIR}'. Please place your VIRAT videos there.")
        return

    print(f"Found {len(video_files)} video files. Starting extraction...")
    for vid_path in tqdm(video_files, desc="Extracting frames"):
        filename = os.path.basename(vid_path)
        task_name = get_task_folder(filename)
        video_stem = os.path.splitext(filename)[0]
        
        # Place frames inside task/default_class to support PyTorch ImageFolder
        out_dir = os.path.join(OUTPUT_BASE_DIR, task_name, "general_activity", video_stem)
        extract_frames_from_video(vid_path, out_dir, target_fps=1)

    print("Frame extraction completed successfully.")

if __name__ == "__main__":
    process_all_videos()