import cv2 as cv
import mediapipe as mp
import numpy as np
import os
import streamlit as st
from PIL import Image
from datetime import datetime



# ---------------- Page Config ----------------
st.set_page_config(
    page_title="Virtual Background Studio",
    layout="wide",
    initial_sidebar_state="collapsed",
    page_icon=":grin:"
)

st.title("🎥 Virtual Background Studio" , text_alignment="center")
st.caption("Real-time background replacement using MediaPipe Selfie Segmentation" , width="stretch" , text_alignment="center")
st.divider()

# ---------------- MediaPipe Setup ----------------
mp_image_segmenter = mp.tasks.vision.ImageSegmenter
mp_base_options = mp.tasks.BaseOptions
mp_vision = mp.tasks.vision

model_path = "./Zoom/Model/selfie_segmenter.tflite"
base_options = mp_base_options(model_asset_path=model_path)

options = mp_vision.ImageSegmenterOptions(
    base_options=base_options,
    running_mode=mp.tasks.vision.RunningMode.VIDEO,
    output_category_mask=True
)

# ---------------- Background Folder ----------------
bg_folder = "./Zoom/backgrounds"
os.makedirs(bg_folder, exist_ok=True)

def load_bg_paths():
    return [
        os.path.join(bg_folder, f)
        for f in os.listdir(bg_folder)
        if f.lower().endswith((".png", ".jpg", ".jpeg"))
    ]

bg_images_paths = load_bg_paths()

# ---------------- Session State ----------------
if "current_bg_img" not in st.session_state:
    st.session_state.current_bg_img = None
if "last_uploaded_name" not in st.session_state:
    st.session_state.last_uploaded_name = None

# ---------------- Layout ----------------
video_col, control_col = st.columns([2, 1], gap="medium")

# ===================== CONTROL PANEL =====================
with control_col:
    # Create bordered container effect
    with st.container(border=True):
        st.header(" Background Controls" , text_alignment="center" , divider=True)
        
        # ---- Upload ----
        uploaded_file = st.file_uploader(
            "Upload Background",
            type=["jpg", "jpeg", "png"],
            help="Upload a custom background image"
        )

        if uploaded_file and uploaded_file.name != st.session_state.last_uploaded_name:
            # Save image so it appears in grid
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_path = os.path.join(bg_folder, f"new_{timestamp}.png")

            image = Image.open(uploaded_file).convert("RGB")
            image.save(save_path)

            st.session_state.current_bg_img = cv.imread(save_path)
            st.session_state.last_uploaded_name = uploaded_file.name
            st.success(" Background uploaded & selected")

            # Refresh grid
            bg_images_paths = load_bg_paths()

        st.divider()

        # ---- Gallery ----
        st.subheader("Available Backgrounds" , width="stretch" , text_alignment="center" , divider=True)
        
        if bg_images_paths:
            # Display images in rows of 2 with equal heights
            for i in range(0, len(bg_images_paths), 2):
                row_imgs = bg_images_paths[i:i+2]
                
                cols = st.columns(2, gap="small")
                
                for col_idx, img_path in enumerate(row_imgs):
                    with cols[col_idx]:
                        # Create bordered container for each image - clickable
                        with st.container(border=True):
                            if st.button(
                                "Select",
                                key=f"bg_{i + col_idx}",
                                width="content",
                                type="secondary"
                            ):
                                st.session_state.current_bg_img = cv.imread(img_path)
                                st.toast("Background selected")
                            
                            # Display image inside the button area
                            st.image(img_path, width="content")
                
                # Add placeholder for odd number of images
                if len(row_imgs) == 1:
                    with cols[1]:
                        with st.container(border=True):
                            st.empty()
                
                # Add divider between rows (except after last row)
                if i + 2 < len(bg_images_paths):
                    st.divider()
        else:
            st.info("No backgrounds yet. Upload one to get started!")

# ===================== VIDEO PANEL =====================
with video_col:
    # Create bordered container effect
    with st.container(border=True , horizontal_alignment="center"):
        st.subheader("Live Camera Feed" , text_alignment="center" , divider=True)
        
        run = st.toggle("Start Camera", value=True)
        
        frame_placeholder = st.empty()

        if run:
            cap = cv.VideoCapture(0)

            with mp_image_segmenter.create_from_options(options) as segmenter:
                while run:
                    ret, frame = cap.read()
                    if not ret:
                        st.error("Camera not detected")
                        break

                    frame = cv.flip(frame, 1)
                    blur = cv.GaussianBlur(frame , (7,7) , 0)
                    frame = cv.addWeighted(frame, 1.8, blur, -0.8, 0)
                    h, w, _ = frame.shape

                    # ---- Background Prep ----
                    if st.session_state.current_bg_img is None:
                        current_bg = np.zeros((h, w, 3), dtype=np.uint8)
                        current_bg[:] = (0, 255, 0)
                    else:
                        current_bg = cv.resize(
                            st.session_state.current_bg_img, (w, h)
                        )

                    # ---- MediaPipe ----
                    mp_image = mp.Image(
                        image_format=mp.ImageFormat.SRGB,
                        data=frame
                    )

                    timestamp_ms = int(
                        cv.getTickCount() * 1000 / cv.getTickFrequency()
                    )

                    result = segmenter.segment_for_video(
                        mp_image, timestamp_ms
                    )

                    category_mask = result.category_mask.numpy_view()

                    # ---- Masking ----
                    person_mask = category_mask > 0.1
                    mask_3ch = np.dstack(
                        (person_mask, person_mask, person_mask)
                    )

                    output = np.where(mask_3ch, current_bg, frame)
                    blurred = cv.GaussianBlur(output, (3,3), 0.8)
                    output = cv.addWeighted(output, 1.8, blurred, -0.8, 0)
                    output = cv.cvtColor(output, cv.COLOR_BGR2RGB)

                    frame_placeholder.image(output, channels="RGB", width="content")

            cap.release()

        else:
            with frame_placeholder.container():
                st.info("**Ready to start!**")
                st.markdown("""
                1. Select a background from the gallery or upload your own
                2. Enable **Start Camera** to begin the magic
                3. See yourself with your chosen background in real-time
                """)

st.divider()

st.caption("Powered by MediaPipe Selfie Segmentation")
