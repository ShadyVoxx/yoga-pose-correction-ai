#!/usr/bin/env python3
"""
download_drive_data.py — Download landmarks.json + metadata.json from the
yoga dataset Google Drive folder and stage them for merge_new_participants.py.

Usage:
    python3 download_drive_data.py [--out ./new_sessions] [--folder FOLDER_ID]

First run: opens a browser for Google OAuth consent. Token saved to token.json.
Subsequent runs: uses saved token automatically.

Skips files already downloaded (checks by size).
Downloads only landmarks.json and metadata.json (skips video/IMU — saves ~90% bandwidth).
"""

import argparse
import base64
import json
import os
import sys
import time
from pathlib import Path

# --------------------------------------------------------------------------
# Google Drive folder IDs to pull from
# --------------------------------------------------------------------------
DRIVE_FOLDERS = {
    # NEW data (not yet in training_data.json)
    "VMCY_July2026":   "1g12i88VkUfcTFERXvEbw_tXy-ftGtpRp",
    "cblyoga_setB":    "1OlRnrovBgzBedUJlDQ11Mc5fV1qAfpwh",
    "vmcy_setB":       "1ACjMlg8KIVMj01d_wsWgDmgSasXCHHvd",
}

DOWNLOAD_FILES = {"landmarks.json", "metadata.json"}

SCOPES = ["https://www.googleapis.com/auth/drive.readonly"]
TOKEN_FILE = Path(__file__).parent / "gdrive_token.json"
CREDS_FILE = Path(__file__).parent / "gdrive_credentials.json"

# ── Auth ─────────────────────────────────────────────────────────────────────
def get_credentials():
    from google.oauth2.credentials import Credentials
    from google.auth.transport.requests import Request
    from google_auth_oauthlib.flow import InstalledAppFlow

    creds = None
    if TOKEN_FILE.exists():
        creds = Credentials.from_authorized_user_file(str(TOKEN_FILE), SCOPES)
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            if not CREDS_FILE.exists():
                print(f"""
❌  {CREDS_FILE} not found.

To set up Google Drive access:
  1. Go to https://console.cloud.google.com/
  2. Create a project → Enable Drive API
  3. Credentials → Create OAuth 2.0 Client ID (Desktop app)
  4. Download the JSON and save it as:
       {CREDS_FILE}
  5. Re-run this script.
""")
                sys.exit(1)
            flow = InstalledAppFlow.from_client_secrets_file(str(CREDS_FILE), SCOPES)
            creds = flow.run_local_server(port=0)
        TOKEN_FILE.write_text(creds.to_json())
    return creds


# ── Drive helpers ─────────────────────────────────────────────────────────────
def list_folder(service, folder_id):
    """Return list of {id, name, mimeType, size} items in a Drive folder."""
    items = []
    page_token = None
    while True:
        q = f"'{folder_id}' in parents and trashed = false"
        resp = service.files().list(
            q=q,
            fields="nextPageToken, files(id, name, mimeType, size)",
            pageToken=page_token,
            pageSize=200,
        ).execute()
        items.extend(resp.get("files", []))
        page_token = resp.get("nextPageToken")
        if not page_token:
            break
    return items


def download_file(service, file_id, dest_path, size=None):
    """Download a Drive file to dest_path."""
    from googleapiclient.http import MediaIoBaseDownload
    import io

    dest_path = Path(dest_path)
    dest_path.parent.mkdir(parents=True, exist_ok=True)

    if dest_path.exists() and size and dest_path.stat().st_size == int(size):
        return False  # already downloaded

    req = service.files().get_media(fileId=file_id)
    buf = io.BytesIO()
    downloader = MediaIoBaseDownload(buf, req)
    done = False
    while not done:
        _, done = downloader.next_chunk()
    dest_path.write_bytes(buf.getvalue())
    return True


def walk_and_download(service, folder_id, local_dir, depth=0, stats=None):
    """Recursively walk a Drive folder, downloading target files."""
    if stats is None:
        stats = {"downloaded": 0, "skipped": 0, "folders": 0}

    items = list_folder(service, folder_id)
    for item in items:
        name = item["name"]
        mime = item["mimeType"]
        if mime == "application/vnd.google-apps.folder":
            stats["folders"] += 1
            walk_and_download(service, item["id"], os.path.join(local_dir, name), depth+1, stats)
        elif name in DOWNLOAD_FILES:
            dest = os.path.join(local_dir, name)
            size = item.get("size")
            downloaded = download_file(service, item["id"], dest, size)
            if downloaded:
                stats["downloaded"] += 1
                print(f"  {'  '*depth}⬇  {name}  ({int(size or 0)//1024}KB)  →  {dest}")
            else:
                stats["skipped"] += 1
    return stats


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Download yoga Drive data")
    parser.add_argument("--out", default="./new_sessions",
                        help="Local staging directory (default: ./new_sessions)")
    parser.add_argument("--folder", default=None,
                        help="Single Drive folder ID to download (default: all)")
    args = parser.parse_args()

    from googleapiclient.discovery import build
    creds = get_credentials()
    service = build("drive", "v3", credentials=creds, cache_discovery=False)

    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)

    total_stats = {"downloaded": 0, "skipped": 0, "folders": 0}

    folders = DRIVE_FOLDERS
    if args.folder:
        folders = {"custom": args.folder}

    for label, folder_id in folders.items():
        print(f"\n📂  {label}  ({folder_id})")
        stats = walk_and_download(service, folder_id, str(out_root / label))
        print(f"    ✅ downloaded={stats['downloaded']}  skipped={stats['skipped']}  folders={stats['folders']}")
        for k in total_stats:
            total_stats[k] += stats[k]

    print(f"\n🎉  Total: {total_stats['downloaded']} files downloaded, {total_stats['skipped']} already present")
    print(f"\nNext step:\n  python3 merge_new_participants.py {out_root} training_data.json")


if __name__ == "__main__":
    main()
