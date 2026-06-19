#!/usr/bin/env python3
"""
prepare_dataset_v2.py — Build training_data.json (v2) from raw recorded pose sessions.

Improvements over prepare_dataset.py:
  1. NORMALIZED LANDMARKS — each frame's 33 (x,y,z) landmarks are translated so
     the hip midpoint is the origin, and scaled by the shoulder-to-hip distance
     (torso length). This removes camera distance/position variation, which is
     one of the biggest sources of noise for a raw-coordinate classifier.
     server.js MUST apply the exact same normalization to live landmarks
     before inference (see normalizeLandmarks() in server.js).

  2. MIRROR AUGMENTATION — every normalized sample is duplicated with a
     left-right mirror flip (x negated, L/R landmark pairs swapped). This
     roughly doubles the dataset and helps the model generalize to either
     side of asymmetric poses (e.g. Trikonasana bending left vs right).

  3. CLUSTERING-BASED STEP LABELS — instead of splitting each recording into
     K equal-time chunks, frames are grouped into K clusters (K = number of
     steps for that pose) via a simple from-scratch k-means on the normalized
     landmark vectors. Clusters are then ordered by their mean frame index
     (temporal centroid) so cluster 0 = earliest, ... cluster K-1 = latest,
     and a light temporal median-filter smoothing pass reduces flicker. This
     groups visually-similar frames together as a "step" rather than assuming
     uniform timing.

IMU features are extracted exactly as in prepare_dataset.py (nearest-timestamp
match), stored for future use, and zero-filled during training/inference while
IMU_ENABLED=false.
"""

import json
import os
import sys
import zipfile
from glob import glob

import numpy as np

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

DOWNSAMPLE_EVERY = 2  # keep every Nth landmark frame (before mirror doubling)

# --------------------------------------------------------------------------
# Landmark normalization / mirroring — MUST match server.js exactly.
# --------------------------------------------------------------------------
LEFT_HIP, RIGHT_HIP = 23, 24
LEFT_SHOULDER, RIGHT_SHOULDER = 11, 12

# (left_index, right_index) pairs for the 33 MediaPipe Pose landmarks
MIRROR_PAIRS = [
    (1, 4), (2, 5), (3, 6), (7, 8), (9, 10),
    (11, 12), (13, 14), (15, 16), (17, 18), (19, 20),
    (21, 22), (23, 24), (25, 26), (27, 28), (29, 30), (31, 32),
]


def normalize_landmarks(flat99):
    """Hip-centered, torso-scale-normalized landmarks. flat99: 99 floats -> 99 floats."""
    pts = np.asarray(flat99, dtype=np.float64).reshape(33, 3)
    hip_mid = (pts[LEFT_HIP] + pts[RIGHT_HIP]) / 2.0
    shoulder_mid = (pts[LEFT_SHOULDER] + pts[RIGHT_SHOULDER]) / 2.0
    scale = float(np.linalg.norm(shoulder_mid - hip_mid))
    if scale < 1e-6:
        scale = 1.0
    norm = (pts - hip_mid) / scale
    return norm.flatten().tolist()


def mirror_landmarks(flat99):
    """Left-right mirror of already-normalized landmarks."""
    pts = np.asarray(flat99, dtype=np.float64).reshape(33, 3).copy()
    pts[:, 0] = -pts[:, 0]
    for a, b in MIRROR_PAIRS:
        pts[[a, b]] = pts[[b, a]]
    return pts.flatten().tolist()


# --------------------------------------------------------------------------
# From-scratch k-means (no sklearn available in sandbox)
# --------------------------------------------------------------------------
def kmeans(X, k, iters=50, seed=0):
    n = X.shape[0]
    if n <= k:
        return np.arange(n) % k
    rng = np.random.RandomState(seed)
    # k-means++ initialization
    centers = X[rng.choice(n, 1)]
    for _ in range(k - 1):
        d2 = np.min(((X[:, None, :] - centers[None, :, :]) ** 2).sum(-1), axis=1)
        total = d2.sum()
        probs = d2 / total if total > 1e-12 else np.full(n, 1.0 / n)
        idx = rng.choice(n, 1, p=probs)
        centers = np.vstack([centers, X[idx]])

    labels = np.zeros(n, dtype=int)
    for _ in range(iters):
        d2 = ((X[:, None, :] - centers[None, :, :]) ** 2).sum(-1)
        new_labels = np.argmin(d2, axis=1)
        if np.array_equal(new_labels, labels) and _ > 0:
            labels = new_labels
            break
        labels = new_labels
        new_centers = np.array([
            X[labels == j].mean(axis=0) if np.any(labels == j) else centers[j]
            for j in range(k)
        ])
        centers = new_centers
    return labels


def median_smooth(arr, window=5):
    """Simple temporal median filter to reduce step-label flicker."""
    n = len(arr)
    out = np.array(arr, dtype=int)
    half = window // 2
    for i in range(n):
        lo, hi = max(0, i - half), min(n, i + half + 1)
        vals, counts = np.unique(out[lo:hi], return_counts=True)
        out[i] = vals[np.argmax(counts)]
    return out


def find_nearest_imu(packets_sorted, ts):
    if not packets_sorted:
        return [0.0] * 6
    lo, hi = 0, len(packets_sorted) - 1
    while lo < hi:
        mid = (lo + hi) // 2
        if packets_sorted[mid][0] < ts:
            lo = mid + 1
        else:
            hi = mid
    best = packets_sorted[lo]
    if lo > 0:
        prev = packets_sorted[lo - 1]
        if abs(prev[0] - ts) < abs(best[0] - ts):
            best = prev
    return best[1]


def process_zip(zip_path):
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
    if len(frames) == 0:
        return entries

    # Downsample first
    sampled = [(idx, f) for idx, f in enumerate(frames) if idx % DOWNSAMPLE_EVERY == 0 and len(f.get("landmarks", [])) == 33]
    if not sampled:
        return entries

    # Build normalized feature matrix for clustering
    norm_feats = []
    for _, frame in sampled:
        flat = []
        for lm in frame["landmarks"]:
            flat.extend([lm["x"], lm["y"], lm["z"]])
        norm_feats.append(normalize_landmarks(flat))
    X = np.array(norm_feats, dtype=np.float64)

    k = min(num_steps, len(sampled))
    cluster_ids = kmeans(X, k)

    # Order clusters by temporal centroid -> step index 0..k-1 (chronological)
    centroids_time = {}
    for c in range(k):
        idxs = np.where(cluster_ids == c)[0]
        centroids_time[c] = idxs.mean() if len(idxs) else 1e18
    order = sorted(range(k), key=lambda c: centroids_time[c])
    cluster_to_step = {c: step for step, c in enumerate(order)}
    step_seq = np.array([cluster_to_step[c] for c in cluster_ids], dtype=int)

    # Smooth to reduce flicker, then clip into 0..num_steps-1
    step_seq = median_smooth(step_seq, window=5)
    step_seq = np.clip(step_seq, 0, num_steps - 1)

    for (idx, frame), norm_flat, step_index in zip(sampled, norm_feats, step_seq):
        imu_features = find_nearest_imu(imu_packets, frame.get("relative_timestamp", 0))
        step_index = int(step_index)

        base = {
            "poseName": pose_name,
            "stepIndex": step_index,
            "label": f"{pose_name}_Step{step_index}",
            "features": [round(f, 6) for f in norm_flat],
            "imuFeatures": [round(f, 6) for f in imu_features],
        }
        entries.append(base)

        # Mirror augmentation
        mirrored = mirror_landmarks(norm_flat)
        entries.append({
            "poseName": pose_name,
            "stepIndex": step_index,
            "label": f"{pose_name}_Step{step_index}",
            "features": [round(f, 6) for f in mirrored],
            "imuFeatures": [round(f, 6) for f in imu_features],
        })

    return entries


def main():
    if len(sys.argv) < 2:
        print("Usage: python3 prepare_dataset_v2.py <dataset_root> [<dataset_root2> ...] [--out training_data.json]")
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
