import shutil
from pathlib import Path

src_root = Path("MURA-v1.1")
dst_root = Path("MURA-v1.1-processed")

dst_root.mkdir(parents=True, exist_ok=True)

shutil.copytree(src_root / "train",
                dst_root / "train",
                dirs_exist_ok=True)

for csv_name in ["train_image_paths.csv", "train_labeled_studies.csv"]:
    shutil.copy2(src_root / csv_name, dst_root / csv_name)

old_prefix = "MURA-v1.1"
new_prefix = "MURA-v1.1-processed"

for csv_path in (dst_root / "train_image_paths.csv", dst_root / "train_labeled_studies.csv"):
    text = csv_path.read_text()
    text = text.replace(old_prefix, new_prefix)
    csv_path.write_text(text)

print("Copy data done:", dst_root)