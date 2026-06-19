#!/usr/bin/env python3
"""
prepare_dataset.py — Build training_data.json from raw recorded pose sessions.

Reads the per-pose zip archives produced by the data-collection app
(each containing landmarks.json, imu_data.json, metadata.json, video.webm)
and turns them into labeled feature vectors for train_model.js.

INPUT  : a root folder containing one subfolder per participant, each with
         "<Session>/...zip" archives named like:
         "..._S01_<POSE-ID>_<Pose_Name>_<timestamp>.zip"

OUTPUT : training_data.json — a flat list of:
         {
           "poseName": "Tadasana (Mountain Pose)",
           "stepIndex": 0,
           "label": "Tadasana (Mountain Pose)_Step0",
           "features": [ ... 99 floats: 33 landmarks * (x, y, z) ... ],
           "imuFeatures": [ ax, ay, az, gx, gy, gz ]   # real values, stored
                                                         # for future use but
                                                         # NOT used unless
                                                         # IMU_ENABLED=true in
                                                         # train_model.js / server.js
         }

STEP LABELING HEURISTIC
------------------------
The raw recordings only capture "this whole clip is pose X" — there is no
ground truth for which frame corresponds to which instructional step
(Step 0, Step 1, ...). To still get *some* step-level signal, each
recording's frames are split chronologically into K equal contiguous
chunks (K = number of steps defined for that pose in server.js POSES),
in order: chunk i -> "<PoseName>_Step{i}".

This is a noisy approximation (real step timing isn't uniform), but it's
transparent, deterministic, and gives every step real — if imperfect —
training signal instead of the 1-2 hand-collected samples that existed
before. It can be refined later with manually-labeled step data via
data_collector.js.
"""

import json
import os
import re
import sys
import zipfile
from glob import glob

# --------------------------------------------------------------------------
# Pose ID -> (name as used in server.js POSES[].name, number of steps)
# Keep this in sync with the POSES array in server.js.
# --------------------------------------------------------------------------
POSE_MAP = {
    "STA-01":    ("Tadasana (Mountain Pose)", 6),
    "STA-02":    ("Vrkasana (Tree Pose)", 5),
    "STA-03":    ("Pada-hastasana (Hand-to-Foot Pose)", 6),
    "STA-04-I":  ("Ardha Cakrasana (Half-Wheel Pose)", 6),
    "STA-04-II": ("Ardha Katichakrasana (Half Waist-Wheel Pose)", 5),
    "STA-05-I":  ("Trikonasana (Triangle Pose)", 7),
    "STA-05-II": ("Parivritta Trikonasana (Revolved Triangle Pose)", 6),
    "SIA-01":    ("Ardha Ustrasana (Half Camel Pose)", 6),
    "SIA-02":    ("Vakrasana (Twisted Pose)", 6),
    "PR-01":     ("Makarasana (Crocodile Pose)", 5),
    "PR-02":     ("Bhujangasana (Cobra Pose)", 6),
    "SU-01":     ("Ardha Halasana (Half Plough Pose)", 5),
    "SU-02":     ("Savasana (Corpse Pose)", 5),
}

DOWNSAMPLE_EVERY = 2  # keep every Nth landmark frame


def find_nearest_imu(packets_sorted, ts):
    """Binary search for the IMU packet with relative_timestamp closest to ts."""
    if not packets_sorted:
        return [0.0] * 6
    lo, hi = 0, len(packets_sorted) - 1
    while lo < hi:
        mid = (lo + hi) // 2
        if packets_sorted[mid][0] < ts:
            lo = mid + 1
        else:
            hi = mid
    # lo is first index with timestamp >= ts; compare with lo-1 too
    best = packets_sorted[lo]
    if lo > 0:
        prev = packets_sorted[lo - 1]
        if abs(prev[0] - ts) < abs(best[0] - ts):
            best = prev
    return best[1]


def process_zip(zip_path):
    """Return a list of training entries extracted from one pose recording zip."""
    entries = []
    try:
        with zipfile.ZipFile(zip_path, "r") as z:
            names = set(z.namelist())
            if "landmarks.json" not in names or "metadata.json" not in names:
                print(f"  ⚠️  skipping (missing landmarks/metadata): {os.path.basename(zip_path)}")
                return entries

            metadata = json.loads(z.read("metadata.json"))
            landmarks_doc = json.loads(z.read("landmarks.json"))

            imu_packets = []
            if "imu_data.json" in names:
                imu_doc = json.loads(z.read("imu_data.json"))
                imu_packets = sorted(
                    (
                        (p["relative_timestamp"], [p["ax"], p["ay"], p["az"], p["gx"], p["gy"], p["gz"]])
                        for p in imu_doc.get("packets", [])
                    ),
                    key=lambda x: x[0],
                )
    except (zipfile.BadZipFile, KeyError, json.JSONDecodeError) as e:
        print(f"  ⚠️  skipping (read error: {e}): {os.path.basename(zip_path)}")
        return entries

    pose_id = metadata.get("poseId")
    if pose_id not in POSE_MAP:
        print(f"  ⚠️  skipping (unmapped poseId '{pose_id}'): {os.path.basename(zip_path)}")
        return entries

    pose_name, num_steps = POSE_MAP[pose_id]
    frames = landmarks_doc.get("landmarks", [])
    total_frames = len(frames)
    if total_frames == 0:
        return entries

    chunk_size = max(1, total_frames // num_steps)

    for idx, frame in enumerate(frames):
        if idx % DOWNSAMPLE_EVERY != 0:
            continue

        lm_list = frame.get("landmarks", [])
        if len(lm_list) != 33:
            continue

        features = []
        for lm in lm_list:
            features.extend([lm["x"], lm["y"], lm["z"]])

        step_index = min(idx // chunk_size, num_steps - 1)

        imu_features = find_nearest_imu(imu_packets, frame.get("relative_timestamp", 0))

        entries.append({
            "poseName": pose_name,
            "stepIndex": step_index,
            "label": f"{pose_name}_Step{step_index}",
            "features": [round(f, 6) for f in features],
            "imuFeatures": [round(f, 6) for f in imu_features],
        })

    return entries


def main():
    if len(sys.argv) < 2:
        print("Usage: python3 prepare_dataset.py <dataset_root> [<dataset_root2> ...] [--out training_data.json]")
        sys.exit(1)

    args = sys.argv[1:]
    out_path = "training_data.json"
    if "--out" in args:
        i = args.index("--out")
        out_path = args[i + 1]
        del args[i:i + 2]

    roots = args
    all_entries = []
    zip_count = 0

    for root in roots:
        zips = sorted(glob(os.path.join(root, "**", "*.zip"), recursive=True))
        print(f"📂 {root}: found {len(zips)} zip archives")
        for zp in zips:
            entries = process_zip(zp)
            if entries:
                zip_count += 1
                print(f"  ✅ {os.path.basename(zp)} -> {len(entries)} samples ({entries[0]['poseName']})")
            all_entries.extend(entries)

    if not all_entries:
        print("❌ No samples extracted.")
        sys.exit(1)

    with open(out_path, "w") as f:
        json.dump(all_entries, f)

    labels = sorted(set(e["label"] for e in all_entries))
    print(f"\n💾 Wrote {len(all_entries)} samples from {zip_count} recordings to {out_path}")
    print(f"📊 {len(labels)} unique step labels across {len(set(e['poseName'] for e in all_entries))} poses")


if __name__ == "__main__":
    main()
