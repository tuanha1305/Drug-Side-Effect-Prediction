#!/usr/bin/env python3
"""
Upload the dataset folder to Hugging Face Hub (as a dataset repo).
Usage:
  HF_TOKEN=hf_xxx python hf_upload.py --repo tuanha1305/drug-side-effect --private False --path /path/to/processed
"""
import argparse, os, sys
from huggingface_hub import HfApi, create_repo, upload_folder

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True, help="eg. tuanha1305/drug-side-effect")
    ap.add_argument("--private", default="False", help="True/False")
    ap.add_argument("--path", default=".", help="Folder to upload (default=current dir)")
    args = ap.parse_args()

    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
    if not token:
        print("ERROR: Please set HF_TOKEN env var (hf_...)")
        sys.exit(1)

    api = HfApi(token=token)
    create_repo(repo_id=args.repo, repo_type="dataset", private=(args.private.lower()=="true"), exist_ok=True)
    print(f"Repo ensured: {args.repo}")

    upload_folder(
        repo_id=args.repo,
        repo_type="dataset",
        folder_path=args.path,
        path_in_repo=".",
        ignore_patterns=[]
    )
    print("Upload complete.")

if __name__ == "__main__":
    main()
