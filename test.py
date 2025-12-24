import cv2
import mediapipe as mp
import numpy as np
import urllib.request
import os
import time  # 1. Added time for FPS

# --- Setup MediaPipe ---
mp_image_segmenter = mp.tasks.vision.ImageSegmenter
mp_base_options = mp.tasks.BaseOptions
mp_vision = mp.tasks.vision

model_path = "./Model/selfie_segmenter.tflite"

# Ensure model exists
if not os.path.exists(model_path):
    print("Model not found. Please check your path.")
    # You might want to uncomment the download logic here if needed

base_options = mp_base_options(model_asset_path=model_path)
options = mp_vision.ImageSegmenterOptions(
    base_options=base_options,
    running_mode=mp.tasks.vision.RunningMode.VIDEO,
    output_category_mask=True
)

# --- Setup Background ---
bg_image_path = "./backgrounds/1.png"
bg_image_list = []
bg_folder = "./backgrounds/"

if os.path.exists(bg_folder):
    files = os.listdir(bg_folder)
    # Filter only for image files (png, jpg, jpeg)
    bg_images_list = [os.path.join(bg_folder, f) for f in files if f.lower().endswith(('.png', '.jpg', '.jpeg'))]    

current_bg_index = 0
use_solid = len(bg_images_list) == 0

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



if os.path.exists(bg_image_path):
    bg_image = cv2.imread(bg_image_path)
    use_solid = False
else:
    print("Background not found — using solid green.")
    use_solid = True

# --- Setup Video & FPS ---
cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

pTime = 0  # Previous time for FPS

with mp_image_segmenter.create_from_options(options) as segmenter:
    while True:
        ret, frame = cap.read()
        if not ret: break
        
        frame = cv2.flip(frame, 1)

        # Process MediaPipe
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
        timestamp_ms = int(cv2.getTickCount() * 1000 / cv2.getTickFrequency())
        result = segmenter.segment_for_video(mp_image, timestamp_ms)

        # --- LOGIC FIX START ---
        category_mask = result.category_mask.numpy_view()
        
        # 1. Create the mask where Person (Indices > 0) is TRUE (1)
        person_mask = (category_mask > 0).astype(np.uint8)
        mask_3ch = cv2.merge([person_mask, person_mask, person_mask])

        # 2. Resize background to match frame
        if use_solid:
            bg_resized = np.full_like(frame, (0, 255, 0))
        else:
            bg_resized = cv2.resize(bg_image, (frame.shape[1], frame.shape[0]))

        # 3. Combine: 
        # WHERE mask is 1 (Person) -> Show FRAME
        # WHERE mask is 0 (Background) -> Show BG_IMAGE
        output = np.where(mask_3ch, bg_resized,frame)
        # --- LOGIC FIX END ---

        # --- FPS Calculation ---
        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime
        

        # Add FPS to the Output Image
        cv2.putText(output, f'FPS: {int(fps)}', (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Stack images (Original | Result)
        out = np.hstack([frame, output])
        
        cv2.imshow("Background Replaced", out)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()