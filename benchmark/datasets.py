#!/usr/bin/env python3
"""Fetch or verify the exact 442 benchmark images listed in datasets.json.

Default: download missing files and verify every stored byte hash. Existing
conflicting files are never overwritten. --verify performs no downloads.
--decoded-report additionally needs Pillow and checks decoded RGB hashes.
Only selected regular archive members are streamed; no archive paths or code
are extracted. Dataset source pages and use restrictions are in the manifest.
"""

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import stat
import tarfile
import tempfile
import urllib.error
import urllib.request
import zipfile

HERE = Path(__file__).resolve().parent


def sha256(path):
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def destination(root, relative):
    parts = PurePosixPath(relative)
    if parts.is_absolute() or ".." in parts.parts or "\\" in relative or not relative.startswith("datasets/reference/"):
        raise ValueError(f"Unsafe destination: {relative}")
    result = root.joinpath(*parts.parts)
    for path in (result, *result.parents):
        if path == root:
            break
        if path.is_symlink():
            raise ValueError(f"Refusing a symlink destination: {path}")
    return result


def matching(path, expected):
    if not path.exists():
        return False
    if not path.is_file() or path.stat().st_size != expected["bytes"] or sha256(path) != expected["sha256"]:
        raise RuntimeError(f"Existing file differs from the pinned bytes; refusing overwrite: {path}")
    return True


def store_stream(stream, path, expected):
    """Bound the write, verify before publishing, and atomically refuse overwrite."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix=path.name + ".", suffix=".part", delete=False) as output:
            temporary = Path(output.name)
            digest, count = hashlib.sha256(), 0
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                count += len(block)
                if count > expected["bytes"]:
                    raise RuntimeError(f"Payload exceeds pinned size: {path}")
                digest.update(block)
                output.write(block)
        if count != expected["bytes"] or digest.hexdigest() != expected["sha256"]:
            raise RuntimeError(f"Payload hash/size differs from manifest: {path}")
        try:
            os.link(temporary, path)
        except FileExistsError:
            matching(path, expected)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def download(root, relative, expected, url):
    path = destination(root, relative)
    if matching(path, expected):
        return path
    print(f"Downloading {url}", flush=True)
    request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 libdebayer-benchmark-datasets/1"})
    for attempt in range(3):
        try:
            with urllib.request.urlopen(request, timeout=60) as stream:
                store_stream(stream, path, expected)
            return path
        except (OSError, urllib.error.URLError):
            if attempt == 2:
                raise


def extract_selected(root, archive, recipe, images):
    if recipe["kind"] == "zip":
        with zipfile.ZipFile(archive) as bundle:
            inventory = {}
            for member in bundle.infolist():
                if member.filename in inventory:
                    raise RuntimeError(f"Duplicate archive member: {member.filename}")
                inventory[member.filename] = member
            password = recipe.get("password", "").encode() or None
            for item in images:
                path = destination(root, item["path"])
                if matching(path, item):
                    continue
                member = inventory[item["archive_member"]]
                kind = stat.S_IFMT(member.external_attr >> 16)
                if member.is_dir() or kind not in (0, stat.S_IFREG) or member.file_size != item["bytes"]:
                    raise RuntimeError(f"Invalid selected ZIP member: {member.filename}")
                with bundle.open(member, pwd=password) as stream:
                    store_stream(stream, path, item)
    elif recipe["kind"] == "tar.gz":
        wanted = {item["archive_member"]: item for item in images}
        seen = set()
        with tarfile.open(archive, "r|gz") as bundle:
            for member in bundle:
                if member.name not in wanted:
                    continue
                if member.name in seen or not member.isfile():
                    raise RuntimeError(f"Invalid or duplicate selected TAR member: {member.name}")
                seen.add(member.name)
                item = wanted[member.name]
                if member.size != item["bytes"]:
                    raise RuntimeError(f"Unexpected member size: {member.name}")
                path = destination(root, item["path"])
                if not matching(path, item):
                    with bundle.extractfile(member) as stream:
                        store_stream(stream, path, item)
        if seen != set(wanted):
            raise RuntimeError(f"Archive lacks selected images: {sorted(set(wanted) - seen)}")
    else:
        raise ValueError(f"Unknown archive kind: {recipe['kind']}")


def verify(root, images, decoded_report=None):
    decoded = []
    if decoded_report:
        from PIL import Image, __version__, features
    for item in images:
        path = destination(root, item["path"])
        if not matching(path, item):
            raise RuntimeError(f"Missing benchmark image: {path}")
        if decoded_report:
            with Image.open(path) as source:
                source.load()
                if [source.height, source.width] != item["shape_hw"] or source.mode != "RGB":
                    raise RuntimeError(f"Unexpected image dimensions/mode: {path}")
                image = source.convert("RGB")
                digest = hashlib.sha256(f"{image.width}x{image.height}:RGB:".encode())
                digest.update(image.tobytes())
            decoded.append({"path": item["path"], "decoded_rgb_sha256": digest.hexdigest(),
                            "matches_recorded_pixels": digest.hexdigest() == item["decoded_rgb_sha256"]})
    if decoded_report:
        report = {"pillow": __version__, "libjpeg": features.version_codec("jpg"),
                  "libjpeg_turbo": features.version_feature("libjpeg_turbo") if features.check_feature("libjpeg_turbo") else None,
                  "all_stored_hashes_match": True, "all_decoded_hashes_match": all(r["matches_recorded_pixels"] for r in decoded),
                  "decoded_hash_recipe": "sha256(f'{width}x{height}:RGB:'.encode() + RGB_uint8_bytes)", "images": decoded}
        decoded_report.parent.mkdir(parents=True, exist_ok=True)
        decoded_report.write_text(json.dumps(report, indent=2) + "\n")
        if not report["all_decoded_hashes_match"]:
            raise RuntimeError(f"Stored files match but decoded pixels differ; check decoder versions in {decoded_report}")
    print(f"Verified {len(images)} exact image files: {dict(Counter(i['dataset'] for i in images))}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=HERE / "datasets.json")
    parser.add_argument("--root", type=Path, default=HERE.parent, help="Parent of the datasets/reference/ layout")
    parser.add_argument("--verify", action="store_true", help="Verify existing images without network access")
    parser.add_argument("--decoded-report", type=Path, help="Also decode RGB, compare pixel hashes, and write decoder provenance")
    parser.add_argument("--workers", type=int, default=4, choices=range(1, 5), help="Parallel direct-image downloads")
    args = parser.parse_args()
    root = args.root.resolve()
    manifest = json.loads(args.manifest.read_text())
    images = manifest["images"]
    if len(images) != manifest["image_count"] or len({i["path"] for i in images}) != len(images):
        raise RuntimeError("Manifest count or unique paths are inconsistent")
    pending = [item for item in images if not matching(destination(root, item["path"]), item)]
    if not args.verify:
        direct = []
        for name, dataset in manifest["datasets"].items():
            selected = [item for item in pending if item["dataset"] == name]
            if not selected:
                continue
            if "archive" in dataset:
                archive = dataset["archive"]
                path = download(root, archive["path"], archive, archive["url"])
                extract_selected(root, path, archive, selected)
            else:
                direct.extend((item, dataset["url_template"].format(filename=PurePosixPath(item["path"]).name)) for item in selected)
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = [pool.submit(download, root, item["path"], item, url) for item, url in direct]
            for future in futures:
                future.result()
    verify(root, images, args.decoded_report)


if __name__ == "__main__":
    main()
