#!/usr/bin/env python3
"""
extract_landmarks.py — Extract MediaPipe Pose landmarks from yoga videos.

Reads a video file (local or Google Drive URL), runs MediaPipe Pose on
sampled frames, and appends labelled 99-feature vectors to training_data.json.

Usage:
  # From Google Drive
  python3 extract_landmarks.py \
    --url "https://drive.google.com/file/d/XXXX/view" \
    --pose "Tadasana (Mountain Pose)" --step 0

  # From local file
  python3 extract_landmarks.py \
    --file video.mp4 \
    --pose "Tadasana (Mountain Pose)" --step 0

  # Custom sampling rate (default: every 5th frame)
  python3 extract_landmarks.py --file video.mp4 --pose "..." --step 0 --every 3
"""

import argparse
import json
import os
import sys
import tempfile

import cv2
import mediapipe as mp


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_FILE = os.path.join(SCRIPT_DIR, "training_data.json")


def download_from_gdrive(url: str) -> str:
    """Download a file from Google Drive and return the local path."""
    import gdown

    # Normalise various Drive URL formats to a direct download ID
    file_id = None
    if "/d/" in url:
        file_id = url.split("/d/")[1].split("/")[0]
    elif "id=" in url:
        file_id = url.split("id=")[1].split("&")[0]

    if not file_id:
        print(f"❌ Could not parse Google Drive file ID from: {url}")
        sys.exit(1)

    dl_url = f"https://drive.google.com/uc?id={file_id}"
    dest = os.path.join(tempfile.gettempdir(), f"yoga_video_{file_id}.mp4")

    if os.path.exists(dest):
        print(f"📁 Using cached download: {dest}")
        return dest

    print(f"⬇️  Downloading from Google Drive (ID: {file_id}) ...")
    gdown.download(dl_url, dest, quiet=False)

    if not os.path.exists(dest) or os.path.getsize(dest) == 0:
        print("❌ Download failed or file is empty.")
        sys.exit(1)

    print(f"✅ Downloaded to {dest}")
    return dest


def extract_landmarks_from_video(
    video_path: str,
    pose_name: str,
    step_index: int,
    every_n: int = 5,
    min_confidence: float = 0.5,
):
    """Run MediaPipe Pose on sampled frames and return training entries."""

    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(
        static_image_mode=False,
        model_complexity=2,          # highest accuracy
        min_detection_confidence=min_confidence,
        min_tracking_confidence=min_confidence,
    )

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"❌ Cannot open video: {video_path}")
        sys.exit(1)

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"🎬 Video: {total_frames} frames, {fps:.1f} FPS, ~{total_frames / max(fps, 1):.1f}s")
    print(f"🏷️  Label: {pose_name}_Step{step_index}")
    print(f"📐 Sampling every {every_n}th frame ...")

    entries = []
    frame_idx = 0
    processed = 0
    skipped = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % every_n == 0:
            # MediaPipe expects RGB
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = pose.process(rgb)

            if result.pose_landmarks:
                features = []
                for lm in result.pose_landmarks.landmark:
                    features.extend([lm.x, lm.y, lm.z])

                if len(features) == 99:
                    entries.append({
                        "poseName": pose_name,
                        "stepIndex": step_index,
                        "label": f"{pose_name}_Step{step_index}",
                        "features": [round(f, 6) for f in features],
                    })
                    processed += 1
                else:
                    skipped += 1
            else:
                skipped += 1

        frame_idx += 1

    cap.release()
    pose.close()

    print(f"\n📊 Results: {processed} samples extracted, {skipped} frames skipped (no pose detected)")
    return entries


def merge_and_save(new_entries: list):
    """Append new entries to the existing training_data.json."""
    existing = []
    if os.path.exists(DATA_FILE):
        try:
            with open(DATA_FILE, "r") as f:
                existing = json.load(f)
        except (json.JSONDecodeError, IOError):
            print("⚠️  Could not read existing data, starting fresh.")

    before = len(existing)
    existing.extend(new_entries)

    with open(DATA_FILE, "w") as f:
        json.dump(existing, f, indent=4)

    print(f"💾 training_data.json: {before} → {len(existing)} samples (+{len(new_entries)} new)")


def main():
    parser = argparse.ArgumentParser(
        description="Extract MediaPipe landmarks from yoga videos for ML training."
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--url", help="Google Drive video URL")
    source.add_argument("--file", help="Local video file path")

    parser.add_argument("--pose", required=True, help='Pose name, e.g. "Tadasana (Mountain Pose)"')
    parser.add_argument("--step", required=True, type=int, help="Step index (0-based)")
    parser.add_argument("--every", type=int, default=5, help="Sample every Nth frame (default: 5)")
    parser.add_argument(
        "--confidence", type=float, default=0.5,
        help="Min MediaPipe detection confidence (default: 0.5)",
    )

    args = parser.parse_args()

    # Resolve video path
    if args.url:
        video_path = download_from_gdrive(args.url)
    else:
        video_path = os.path.abspath(args.file)
        if not os.path.exists(video_path):
            print(f"❌ File not found: {video_path}")
            sys.exit(1)

    # Extract
    entries = extract_landmarks_from_video(
        video_path=video_path,
        pose_name=args.pose,
        step_index=args.step,
        every_n=args.every,
        min_confidence=args.confidence,
    )

    if not entries:
        print("❌ No landmarks were extracted. Check that the video shows a person clearly.")
        sys.exit(1)

    # Merge
    merge_and_save(entries)
    print(f"\n✅ Done! Added {len(entries)} samples for \"{args.pose}_Step{args.step}\"")
    print("   Next: run `node train_model.js` to retrain the model.")


if __name__ == "__main__":
    main()
