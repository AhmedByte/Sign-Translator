# preprocess_dataset_batch_mp.py
import os
import cv2
from hand_cropper_batch_mp import get_hand_crop_batch
from concurrent.futures import ProcessPoolExecutor, as_completed

INPUT_ROOT = "/home/abdrabo/Desktop/graduation_project/simulation_data"
OUTPUT_ROOT = "/home/abdrabo/Desktop/graduation_project/sim_cropped_data"
IMG_SIZE = 224
BATCH_SIZE = 16   # عدد الصور لكل batch
NUM_WORKERS = 4   # عدد العمليات المتوازية (حسب CPU)

os.makedirs(OUTPUT_ROOT, exist_ok=True)

def process_batch(batch_imgs, batch_names, save_dir):
    cropped_list = get_hand_crop_batch(batch_imgs)
    for img, cropped, name in zip(batch_imgs, cropped_list, batch_names):
        save_path = os.path.join(save_dir, name)
        if img is None:
            continue
        if cropped is not None and cropped.size > 0:
            final_img = cv2.resize(cropped, (IMG_SIZE, IMG_SIZE))
        else:
            final_img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        cv2.imwrite(save_path, final_img)

def process_folder(class_id, source_id):
    source_path = os.path.join(INPUT_ROOT, class_id, source_id)
    save_dir = os.path.join(OUTPUT_ROOT, class_id, source_id)
    os.makedirs(save_dir, exist_ok=True)

    img_names = sorted(os.listdir(source_path))
    total_imgs = len(img_names)
    print(f"📂 Processing Class {class_id} | Source {source_id} | Total images: {total_imgs}")

    # Create batches
    batches = []
    for i in range(0, total_imgs, BATCH_SIZE):
        batch_names = img_names[i:i+BATCH_SIZE]
        batch_imgs = []
        for name in batch_names:
            path = os.path.join(source_path, name)
            img = cv2.imread(path)
            batch_imgs.append(img)
        batches.append((batch_imgs, batch_names))

    # Process batches in parallel
    with ProcessPoolExecutor(max_workers=NUM_WORKERS) as executor:
        futures = [executor.submit(process_batch, imgs, names, save_dir) for imgs, names in batches]
        for f in as_completed(futures):
            f.result()  # لتأكيد إنه خلص batch

    print(f"✅ Finished Class {class_id} | Source {source_id}")

def main():
    for class_id in sorted(os.listdir(INPUT_ROOT)):
        class_path = os.path.join(INPUT_ROOT, class_id)
        if not os.path.isdir(class_path):
            continue
        for source_id in sorted(os.listdir(class_path)):
            source_path = os.path.join(class_path, source_id)
            if not os.path.isdir(source_path):
                continue
            process_folder(class_id, source_id)

    print("🎉 All dataset preprocessing completed super fast!")

if __name__ == "__main__":
    main()
