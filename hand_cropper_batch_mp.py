# hand_cropper_batch_mp.py
import cv2
import mediapipe as mp
import numpy as np

mp_hands = mp.solutions.hands

def get_hand_crop_batch(images, padding_ratio=0.15):
    """
    Detect hands for a batch of images and return cropped images list.
    images: list of np.array
    Returns: list of cropped images (None if no hand detected)
    """
    cropped_images = []
    
    with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.5
    ) as hands:

        for img in images:
            h, w, _ = img.shape
            results = hands.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

            if not results.multi_hand_landmarks:
                cropped_images.append(None)
                continue

            x_min, y_min = w, h
            x_max, y_max = 0, 0

            for hand_landmarks in results.multi_hand_landmarks:
                for lm in hand_landmarks.landmark:
                    x, y = int(lm.x * w), int(lm.y * h)
                    x_min, y_min = min(x_min, x), min(y_min, y)
                    x_max, y_max = max(x_max, x), max(y_max, y)

            pad_w = int((x_max - x_min) * padding_ratio)
            pad_h = int((y_max - y_min) * padding_ratio)

            x_min = max(0, x_min - pad_w)
            y_min = max(0, y_min - pad_h)
            x_max = min(w, x_max + pad_w)
            y_max = min(h, y_max + pad_h)

            cropped_images.append(img[y_min:y_max, x_min:x_max])

    return cropped_images
