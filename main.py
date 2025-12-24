import cv2
import mediapipe as mp
import numpy as np
import os
import time

# --- Setup MediaPipe ---
mp_image_segmenter = mp.tasks.vision.ImageSegmenter
mp_base_options = mp.tasks.BaseOptions
mp_vision = mp.tasks.vision

model_path = "./Model/selfie_segmenter.tflite"
base_options = mp_base_options(model_asset_path=model_path)
options = mp_vision.ImageSegmenterOptions(
    base_options=base_options,
    running_mode=mp.tasks.vision.RunningMode.VIDEO,
    output_category_mask=True
)

# --- Background Loader System ---
bg_folder = "./backgrounds"
bg_images_list = []

# 1. Load all valid image paths into a list
if os.path.exists(bg_folder):
    files = os.listdir(bg_folder)
    # Filter only for image files (png, jpg, jpeg)
    bg_images_list = [os.path.join(bg_folder, f) for f in files if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

# If folder is empty or doesn't exist, we rely on solid color
current_bg_index = 0
use_solid = len(bg_images_list) == 0

# Helper function to load the current background
def load_background(index, width, height):
    if use_solid:
        # Return Green Screen if no images
        bg = np.zeros((height, width, 3), dtype=np.uint8)
        bg[:] = (0, 255, 0)
        return bg
    else:
        path = bg_images_list[index]
        img = cv2.imread(path)
        return cv2.resize(img, (width, height))

# --- Setup Video ---
cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

# Initial Background Load
current_bg = load_background(current_bg_index, 640, 480)
pTime = 0

with mp_image_segmenter.create_from_options(options) as segmenter:
    while True:
        ret, frame = cap.read()
        if not ret: break
        
        frame = cv2.flip(frame, 1)
        h, w, c = frame.shape

        # MediaPipe Processing
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
        timestamp_ms = int(cv2.getTickCount() * 1000 / cv2.getTickFrequency())
        result = segmenter.segment_for_video(mp_image, timestamp_ms)

        category_mask = result.category_mask.numpy_view()
        
        # Mask Logic (Person > 0)
        person_mask = (category_mask > 0).astype(np.uint8)
        mask_3ch = cv2.merge([person_mask, person_mask, person_mask])

        # Combine
        output = np.where(mask_3ch, current_bg , frame)

        # FPS
        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime
        cv2.putText(output, f'FPS: {int(fps)}', (20, 50), cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 3)

        # Show Result
        out = np.hstack((frame, output))
        cv2.imshow("Zoom Clone - Press 'D' to Switch BG", out)
        
        # --- KEY CONTROLS ---
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q'):
            break
        
        # 2. Switch Background Logic
        elif key == ord('d'):
            if not use_solid:
                # Increment index and wrap around using Modulo (%)
                current_bg_index = (current_bg_index + 1) % len(bg_images_list)
                # Load the new image immediately
                current_bg = load_background(current_bg_index, w, h)
                print(f"Switched to: {bg_images_list[current_bg_index]}")

cap.release()
cv2.destroyAllWindows()