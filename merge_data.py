import os
import shutil

BASE_ROOT = "/home/abdrabo/Desktop/graduation_project"
DST_ROOT = os.path.join(BASE_ROOT, "data")

os.makedirs(DST_ROOT, exist_ok=True)

# sources من 00 ل 17
for i in range(2):
    source_id = f"{i:02d}"   # 00, 01, ..., 17

    source_path = os.path.join(
        BASE_ROOT,
        source_id,
        "Volumes/SarahAlyami/isharah500",
        source_id
    )

    if not os.path.isdir(source_path):
        print(f"⚠️ Skipping {source_id} (not found)")
        continue

    print(f"📂 Processing source {source_id}")

    for class_folder in os.listdir(source_path):
        class_path = os.path.join(source_path, class_folder)
        if not os.path.isdir(class_path):
            continue

        # 01_0003 → 0003
        parts = class_folder.split("_")
        if len(parts) != 2:
            continue

        class_id = parts[1]

        target_dir = os.path.join(DST_ROOT, class_id, source_id)
        os.makedirs(target_dir, exist_ok=True)

        for img in os.listdir(class_path):
            src_img = os.path.join(class_path, img)
            dst_img = os.path.join(target_dir, img)

            if os.path.exists(dst_img):
                name, ext = os.path.splitext(img)
                dst_img = os.path.join(target_dir, f"{name}_{source_id}{ext}")

            shutil.copy(src_img, dst_img)

print("✅ Merge finished for sources 00 → 17")
