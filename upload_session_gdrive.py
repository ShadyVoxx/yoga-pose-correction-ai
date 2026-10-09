#!/usr/bin/env python3
"""
upload_session_gdrive.py — Upload a local YogaDataset session folder to Google Drive.

Usage:
    python3 upload_session_gdrive.py --dir ./YogaDataset/participant1_session1
    python3 upload_session_gdrive.py --auth    # first-time OAuth only

Credentials: gdrive_credentials.json (same file used by download_drive_data.py)
Token saved:  gdrive_token.json (reused on subsequent runs)
"""

import argparse
import os
import sys
from pathlib import Path

SCOPES = ["https://www.googleapis.com/auth/drive"]
TOKEN_FILE = Path(__file__).parent / "gdrive_token.json"
CREDS_FILE = Path(__file__).parent / "gdrive_credentials.json"

# Google Drive folder where sessions are uploaded
GDRIVE_ROOT_FOLDER_NAME = "YogaDataset"


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
                print(f"❌  {CREDS_FILE} not found. Place gdrive_credentials.json in the project root.")
                sys.exit(1)
            flow = InstalledAppFlow.from_client_secrets_file(str(CREDS_FILE), SCOPES)
            creds = flow.run_local_server(port=0)
        TOKEN_FILE.write_text(creds.to_json())
    return creds


def get_or_create_folder(service, name, parent_id=None):
    """Return folder ID, creating it if it doesn't exist."""
    q = f"name='{name}' and mimeType='application/vnd.google-apps.folder' and trashed=false"
    if parent_id:
        q += f" and '{parent_id}' in parents"
    resp = service.files().list(q=q, fields="files(id, name)").execute()
    files = resp.get("files", [])
    if files:
        return files[0]["id"]
    meta = {"name": name, "mimeType": "application/vnd.google-apps.folder"}
    if parent_id:
        meta["parents"] = [parent_id]
    f = service.files().create(body=meta, fields="id").execute()
    return f["id"]


def upload_file(service, local_path, parent_id):
    """Upload a single file, skip if already uploaded (by name + size)."""
    from googleapiclient.http import MediaFileUpload

    name = os.path.basename(local_path)
    size = os.path.getsize(local_path)

    # Check if file already exists with same size
    q = f"name='{name}' and '{parent_id}' in parents and trashed=false"
    resp = service.files().list(q=q, fields="files(id, size)").execute()
    for existing in resp.get("files", []):
        if existing.get("size") and int(existing["size"]) == size:
            print(f"  ⏭  {name} (already uploaded)")
            return

    media = MediaFileUpload(local_path, resumable=True)
    service.files().create(
        body={"name": name, "parents": [parent_id]},
        media_body=media,
        fields="id",
    ).execute()
    print(f"  ⬆  {name}  ({size // 1024}KB)")


def upload_directory(service, local_dir, parent_id):
    """Recursively upload a local directory tree to Drive."""
    local_dir = Path(local_dir)
    folder_id = get_or_create_folder(service, local_dir.name, parent_id)
    for item in sorted(local_dir.iterdir()):
        if item.is_dir():
            upload_directory(service, item, folder_id)
        else:
            upload_file(service, str(item), folder_id)
    return folder_id


def main():
    parser = argparse.ArgumentParser(description="Upload yoga session to Google Drive")
    parser.add_argument("--dir", default=None, help="Local session directory to upload")
    parser.add_argument("--auth", action="store_true", help="Run OAuth only, then exit")
    args = parser.parse_args()

    from googleapiclient.discovery import build
    creds = get_credentials()

    if args.auth:
        print("✅  Authorization complete. Token saved to gdrive_token.json")
        return

    service = build("drive", "v3", credentials=creds, cache_discovery=False)

    # Ensure root YogaDataset folder exists on Drive
    root_id = get_or_create_folder(service, GDRIVE_ROOT_FOLDER_NAME)
    print(f"📂  Google Drive: {GDRIVE_ROOT_FOLDER_NAME} ({root_id})")

    target = Path(args.dir) if args.dir else Path(__file__).parent / "YogaDataset"
    if not target.exists():
        print(f"❌  Directory not found: {target}")
        sys.exit(1)

    print(f"⬆  Uploading: {target}")
    if target.name == "YogaDataset":
        # Upload all sessions inside
        for child in sorted(target.iterdir()):
            if child.is_dir():
                print(f"\n📁  {child.name}")
                upload_directory(service, child, root_id)
    else:
        upload_directory(service, target, root_id)

    print("\n✅  Upload complete.")


if __name__ == "__main__":
    main()
