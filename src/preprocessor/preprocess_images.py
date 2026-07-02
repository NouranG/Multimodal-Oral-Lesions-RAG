"""
preprocess_images.py
Preprocesses all images listed in metadata.csv before CLIP embedding.
Must be run BEFORE build_image_index.py.

Steps applied to each image:
    1. Verify file exists and is a valid image
    2. Convert to RGB (handles RGBA PNGs, grayscale, CMYK)
    3. Resize: long edge capped at 1024px (preserves aspect ratio)
       — CLIP internally resizes to 224x224 anyway, but oversized images
         slow down preprocessing and waste memory
    4. Normalize filename: lowercase, spaces → underscores, remove special chars
    5. Save as JPEG (quality=90) to standardize format
    6. Log any skipped/failed images to preprocess_errors.log

Output:
    - Processed images saved in-place (overwrites originals)
    - preprocess_report.csv: one row per image with status and any issues
"""

import os
import csv
import re
import logging
from PIL import Image, UnidentifiedImageError

# Paths
METADATA_CSV = "data/images/metadata.csv"
REPORT_CSV = "data/images/preprocess_report.csv"
LOG_FILE = "data/images/preprocess_errors.log"

# Config
MAX_LONG_EDGE = 1024   # pixels — cap before CLIP embedding
JPEG_QUALITY = 90
VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}

logging.basicConfig(
    filename=LOG_FILE,
    level=logging.WARNING,
    format="%(asctime)s — %(levelname)s — %(message)s"
)


def normalize_filename(path: str) -> str:
    """
    Ensure filename is clean:
    - lowercase
    - spaces → underscores
    - remove special characters except underscores, hyphens, dots
    - force .jpg extension
    """
    directory = os.path.dirname(path)
    filename = os.path.basename(path)
    name, _ = os.path.splitext(filename)
    name = name.lower()
    name = name.replace(" ", "_")
    name = re.sub(r"[^\w\-]", "", name)   # keep word chars, hyphens only
    return os.path.join(directory, name + ".jpg")


def resize_if_needed(image: Image.Image, max_long_edge: int) -> Image.Image:
    """
    Resize image so its longest edge = max_long_edge.
    If image is already smaller, returns unchanged.
    Preserves aspect ratio using LANCZOS resampling.
    """
    w, h = image.size
    long_edge = max(w, h)
    if long_edge <= max_long_edge:
        return image
    scale = max_long_edge / long_edge
    new_w = int(w * scale)
    new_h = int(h * scale)
    return image.resize((new_w, new_h), Image.LANCZOS)


def preprocess_image(image_path: str) -> dict:
    """
    Process a single image. Returns a status dict for the report.
    """
    status = {
        "original_path": image_path,
        "final_path": image_path,
        "original_size": None,
        "final_size": None,
        "original_mode": None,
        "status": None,
        "issue": ""
    }

    # 1. Check file exists
    if not os.path.exists(image_path):
        status["status"] = "SKIP — file not found"
        logging.warning(f"File not found: {image_path}")
        return status

    # 2. Check extension
    ext = os.path.splitext(image_path)[1].lower()
    if ext not in VALID_EXTENSIONS:
        status["status"] = f"SKIP — unsupported extension: {ext}"
        logging.warning(f"Unsupported extension: {image_path}")
        return status

    # 3. Open and validate
    try:
        image = Image.open(image_path)
        image.verify()              # catches truncated/corrupt files
        image = Image.open(image_path)  # reopen after verify (verify closes the file)
    except UnidentifiedImageError:
        status["status"] = "SKIP — unidentified image format"
        logging.warning(f"Unidentified image: {image_path}")
        return status
    except Exception as e:
        status["status"] = f"SKIP — open error: {e}"
        logging.warning(f"Open error {image_path}: {e}")
        return status

    status["original_size"] = image.size
    status["original_mode"] = image.mode

    # 4. Convert to RGB
    # RGBA: PNG with transparency — flatten onto white background
    if image.mode == "RGBA":
        background = Image.new("RGB", image.size, (255, 255, 255))
        background.paste(image, mask=image.split()[3])  # use alpha channel as mask
        image = background
    # Grayscale, CMYK, or anything else → RGB
    elif image.mode != "RGB":
        image = image.convert("RGB")

    # 5. Resize
    image = resize_if_needed(image, MAX_LONG_EDGE)
    status["final_size"] = image.size

    # 6. Normalize filename and save as JPEG
    final_path = normalize_filename(image_path)
    status["final_path"] = final_path

    os.makedirs(os.path.dirname(final_path), exist_ok=True)
    image.save(final_path, "JPEG", quality=JPEG_QUALITY)

    # Remove original if filename changed
    if final_path != image_path and os.path.exists(image_path):
        os.remove(image_path)

    status["status"] = "OK"
    return status


def update_metadata_paths(metadata_path: str, path_map: dict):
    """
    If any filenames were normalized (path changed), update metadata.csv
    so image_path column stays consistent with actual files on disk.
    """
    with open(metadata_path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    fieldnames = rows[0].keys() if rows else []
    updated = 0
    for row in rows:
        original = row["image_path"]
        if original in path_map and path_map[original] != original:
            row["image_path"] = path_map[original]
            updated += 1

    with open(metadata_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    if updated:
        print(f"  Updated {updated} paths in metadata.csv")


def run_preprocessing():
    print("Loading metadata...")
    with open(METADATA_CSV, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    image_paths = [row["image_path"] for row in rows]
    # Deduplicate — same image may appear in multiple pairs
    unique_paths = list(dict.fromkeys(image_paths))
    print(f"Unique images to process: {len(unique_paths)}")

    report = []
    path_map = {}   # original_path → final_path (for metadata update)
    ok, skipped, errors = 0, 0, 0

    for i, path in enumerate(unique_paths):
        result = preprocess_image(path)
        report.append(result)
        path_map[result["original_path"]] = result["final_path"]

        if result["status"] == "OK":
            ok += 1
        elif "SKIP" in result["status"]:
            skipped += 1
        else:
            errors += 1

        if (i + 1) % 10 == 0:
            print(f"  Processed {i + 1}/{len(unique_paths)}...")

    # Save report
    with open(REPORT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=report[0].keys())
        writer.writeheader()
        writer.writerows(report)

    # Update metadata if any paths changed
    update_metadata_paths(METADATA_CSV, path_map)

    print(f"\nPreprocessing complete:")
    print(f"  OK:      {ok}")
    print(f"  Skipped: {skipped}  (see {LOG_FILE})")
    print(f"  Errors:  {errors}")
    print(f"  Report:  {REPORT_CSV}")

    if skipped > 0 or errors > 0:
        print(f"\nWarning: {skipped + errors} images were skipped.")
        print("Check preprocess_errors.log and collect replacements before building the index.")
    else:
        print("\nAll images processed successfully. Ready to run build_image_index.py")


if __name__ == "__main__":
    run_preprocessing()
