import os
import csv
import hashlib
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image


def md5_bytes(data: bytes) -> str:
    return hashlib.md5(data).hexdigest()


def verify(mapping_csv: str, images_dir: str, check_md5: bool = True, limit: Optional[int] = None) -> None:
    mapping_csv = str(Path(mapping_csv))
    images_dir = str(Path(images_dir))

    missing = 0
    md5_mismatch = 0
    total = 0

    with open(mapping_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        has_md5 = "md5" in reader.fieldnames

        for row in reader:
            total += 1
            if limit is not None and total > limit:
                break

            filename = row["filename"]
            expected_md5 = row.get("md5")

            # If you exported into class folders, this would need adjustment.
            img_path = os.path.join(images_dir, filename)

            if not os.path.exists(img_path):
                missing += 1
                continue

            if check_md5 and has_md5 and expected_md5:
                img = Image.open(img_path)
                arr = np.array(img, dtype=np.uint8)
                got_md5 = md5_bytes(arr.tobytes())
                if got_md5 != expected_md5:
                    md5_mismatch += 1

    print(f"Checked rows: {total}")
    print(f"Missing images: {missing}")
    if check_md5:
        print(f"MD5 mismatches: {md5_mismatch}")
    print("[OK]" if (missing == 0 and (not check_md5 or md5_mismatch == 0)) else "[WARN] Issues found")


if __name__ == "__main__":
    # CHANGE THESE PATHS:
    mapping = r"./cifar10_raw/train/mapping_train.csv"
    images  = r"./cifar10_raw/train"

    verify(mapping, images, check_md5=True)
