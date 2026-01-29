#!/usr/bin/env python3
import argparse
import json
import os
import re
import sys
import tarfile
import urllib.request


FIGSHARE_DOI = "https://doi.org/10.6084/m9.figshare.31180054"
FIGSHARE_ARTICLE_ID = "31180054"


def fetch_json(url):
    with urllib.request.urlopen(url) as response:
        data = response.read().decode("utf-8")
    return json.loads(data)


def normalize_article_id(value):
    if not value:
        return None
    if value.isdigit():
        return value
    match = re.search(r"figshare\.(\d+)", value)
    if match:
        return match.group(1)
    match = re.search(r"/(\d+)(?:\b|$)", value)
    if match:
        return match.group(1)
    return value


def download_file(url, dest_path):
    tmp_path = dest_path + ".part"
    with urllib.request.urlopen(url) as response, open(tmp_path, "wb") as handle:
        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            handle.write(chunk)
    os.replace(tmp_path, dest_path)


def extract_tarball(tar_path, dest_dir):
    with tarfile.open(tar_path, "r:*") as archive:
        archive.extractall(dest_dir)


def classify_dest_dir(filename, model_dir, dataset_dir, default_dir):
    lower = filename.lower()
    if "model" in lower:
        return model_dir
    if "data" in lower or "dataset" in lower:
        return dataset_dir
    return default_dir


def download_figshare_article(
    article_id, model_dir, dataset_dir, default_dir, extract, list_only
):
    files_url = "https://api.figshare.com/v2/articles/{}/files".format(article_id)
    files = fetch_json(files_url)
    if not files:
        print("No files found for article {}".format(article_id))
        return

    for info in files:
        name = info.get("name") or "unknown"
        download_url = info.get("download_url")
        if not download_url:
            print("Skipping {}: no download_url".format(name))
            continue

        dest_dir = classify_dest_dir(name, model_dir, dataset_dir, default_dir)
        dest_path = os.path.join(dest_dir, name)
        if list_only:
            print("{} -> {}".format(name, dest_path))
            continue

        os.makedirs(dest_dir, exist_ok=True)
        print("Downloading {} -> {}".format(name, dest_path))
        download_file(download_url, dest_path)

        if extract and name.endswith(".tar.gz"):
            print("Extracting {} -> {}".format(name, dest_dir))
            extract_tarball(dest_path, dest_dir)


def main():
    parser = argparse.ArgumentParser(
        description="Download model/dataset artifacts from Figshare."
    )
    parser.add_argument(
        "--article",
        default=os.environ.get("FIGSHARE_ARTICLE_ID") or FIGSHARE_ARTICLE_ID,
        help="Figshare article ID or DOI/URL",
    )
    parser.add_argument(
        "--model-dir",
        default="model",
        help="Destination directory for model files",
    )
    parser.add_argument(
        "--dataset-dir",
        default="dataset",
        help="Destination directory for dataset files",
    )
    parser.add_argument(
        "--default-dir",
        default=".",
        help="Fallback directory for unclassified files",
    )
    parser.add_argument(
        "--extract",
        action="store_true",
        help="Extract tar.gz files after download",
    )
    parser.add_argument(
        "--list-only",
        action="store_true",
        help="List planned downloads without fetching",
    )
    args = parser.parse_args()

    article_id = normalize_article_id(args.article)
    if not article_id:
        print("Provide a Figshare article ID or DOI/URL.")
        return 1

    download_figshare_article(
        article_id,
        args.model_dir,
        args.dataset_dir,
        args.default_dir,
        args.extract,
        args.list_only,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
