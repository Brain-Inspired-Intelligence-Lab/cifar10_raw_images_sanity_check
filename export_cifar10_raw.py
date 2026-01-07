import os
import csv
import pickle
import hashlib
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from PIL import Image


def unpickle(file_path: str) -> Dict:
    with open(file_path, "rb") as fo:
        # CIFAR10 python version uses bytes keys by default
        d = pickle.load(fo, encoding="bytes")
    return d


def load_label_names(cifar_dir: str) -> List[str]:
    meta_path = os.path.join(cifar_dir, "batches.meta")
    meta = unpickle(meta_path)
    # label_names stored under b"label_names" (list of bytes)
    names = [x.decode("utf-8") for x in meta[b"label_names"]]
    return names


def md5_bytes(data: bytes) -> str:
    return hashlib.md5(data).hexdigest()


def export_split(
    cifar_dir: str,
    out_dir: str,
    split: str = "train",
    make_class_folders: bool = False,
    add_md5: bool = True,
) -> None:
    """
    Exports CIFAR-10 from the original python batches into PNGs.

    Filenames:
      train: {global_index:05d}_{classId}_{className}.png   (global_index 0..49999)
      test : {index:05d}_{classId}_{className}.png          (index 0..9999)

    Also writes mapping_{split}.csv:
      index, filename, class_id, class_name, batch_file, batch_row, md5(optional)
    """
    assert split in {"train", "test"}

    cifar_dir = str(Path(cifar_dir))
    out_dir = str(Path(out_dir))
    os.makedirs(out_dir, exist_ok=True)

    label_names = load_label_names(cifar_dir)

    if split == "train":
        batch_files = [f"data_batch_{i}" for i in range(1, 6)]
        mapping_csv = os.path.join(out_dir, "mapping_train.csv")
    else:
        batch_files = ["test_batch"]
        mapping_csv = os.path.join(out_dir, "mapping_test.csv")

    global_index = 0

    with open(mapping_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        header = ["index", "filename", "class_id", "class_name", "batch_file", "batch_row"]
        if add_md5:
            header.append("md5")
        writer.writerow(header)

        for bf in batch_files:
            batch_path = os.path.join(cifar_dir, bf)
            d = unpickle(batch_path)

            # data shape: (N, 3072) where 3072 = 3 * 32 * 32
            data = d[b"data"]  # numpy array uint8
            labels = d.get(b"labels", d.get(b"fine_labels"))  # train/test uses b"labels"

            data = np.array(data, dtype=np.uint8)
            labels = np.array(labels, dtype=np.int64)

            # For each row in this batch
            for row_i in range(data.shape[0]):
                y = int(labels[row_i])
                cname = label_names[y]

                # Convert row -> image: reshape to (3,32,32) then transpose -> (32,32,3)
                img_arr = data[row_i].reshape(3, 32, 32).transpose(1, 2, 0)

                # Filename uses a stable index
                idx_str = f"{global_index:05d}"
                filename = f"{idx_str}_{y}_{cname}.png"

                # Optionally write into class folders
                if make_class_folders:
                    class_dir = os.path.join(out_dir, f"{y}_{cname}")
                    os.makedirs(class_dir, exist_ok=True)
                    save_path = os.path.join(class_dir, filename)
                else:
                    save_path = os.path.join(out_dir, filename)

                # Save PNG
                Image.fromarray(img_arr).save(save_path)

                # Optional hash (use raw bytes of array for consistency)
                md5 = md5_bytes(img_arr.tobytes()) if add_md5 else None

                row = [global_index, filename, y, cname, bf, row_i]
                if add_md5:
                    row.append(md5)
                writer.writerow(row)

                global_index += 1

    print(f"[OK] Exported split={split}")
    print(f"     Images -> {out_dir}")
    print(f"     Mapping -> {mapping_csv}")
    if split == "train" and global_index != 50000:
        print(f"[WARN] Expected 50000 train images, got {global_index}")
    if split == "test" and global_index != 10000:
        print(f"[WARN] Expected 10000 test images, got {global_index}")


if __name__ == "__main__":
    # CHANGE THESE PATHS:
    CIFAR_DIR = r"./cifar-10-batches-py"   # folder containing data_batch_1..5, test_batch, batches.meta
    OUT_ROOT = r"./cifar10_raw"

    export_split(CIFAR_DIR, os.path.join(OUT_ROOT, "train"), split="train",
                 make_class_folders=False, add_md5=True)

    export_split(CIFAR_DIR, os.path.join(OUT_ROOT, "test"), split="test",
                 make_class_folders=False, add_md5=True)
