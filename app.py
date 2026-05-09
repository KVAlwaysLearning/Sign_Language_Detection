import streamlit as st
import cv2
import os
import gdown
from ultralytics import YOLO
from datetime import datetime
from PIL import Image
from collections import Counter
import tempfile

# --- CONFIGURATION & DRIVE IDs ---
# Replace these with your actual Google Drive File IDs
ALPHABET_MODEL_ID = "1Cwrlihua2N9Z-2W_RxyV-KzCPFSwplw6"
WORD_MODEL_ID = "1ON3LrBqyBCsW7k6kk35IDbujs-k8CXBK"

ALPHABET_MODEL_PATH = 'best_image_alphabet.pt'
WORD_MODEL_PATH = 'best_video_words.pt'

# --- STARTUP LOOP: DOWNLOAD MODELS ---
def download_models():
    """Checks for existing models and downloads from GDrive if missing."""
    models_to_download = [
        (ALPHABET_MODEL_ID, ALPHABET_MODEL_PATH),
        (WORD_MODEL_ID, WORD_MODEL_PATH)
    ]
    
    for file_id, output_path in models_to_download:
        if not os.path.exists(output_path):
            with st.spinner(f"Downloading {output_path} from Google Drive..."):
                url = f'https://drive.google.com/uc?id={file_id}'
                gdown.download(url, output_path, quiet=False)
            st.success(f"Downloaded {output_path}")

# Run the download logic before anything else
download_models()

# --- MODEL LOADING ---
@st.cache_resource
def load_alphabet_model():
    return YOLO(ALPHABET_MODEL_PATH)

@st.cache_resource
def load_word_model():
    return YOLO(WORD_MODEL_PATH)

# --- APP UI ---
st.set_page_config(page_title="ASL Recognition System", layout="wide")
st.title("🤟 ASL Sign Language Recognition")

# Operational Hours Check (6 PM - 10 PM)
def is_operational():
    current_hour = datetime.now().hour
    return 0 <= current_hour < 24

if not is_operational():
    st.error(f"🛑 Operational hours: 6:00 PM - 10:00 PM. Current: {datetime.now().strftime('%H:%M')}")
    st.stop()

# Tabs for Alphabet/Word detection
tab1, tab2, tab3 = st.tabs(["🔤 Alphabet (Image)", "📝 Word (Image)", "🎥 Word (Video)"])

# Logic for Tab 1 (Alphabet)
with tab1:
    uploaded_alphabet = st.file_uploader("Alphabet Image", type=['jpg', 'jpeg', 'png'], key="alpha")
    if uploaded_alphabet:
        img = Image.open(uploaded_alphabet)
        st.image(img, width=300)
        if st.button("Identify Alphabet"):
            model = load_alphabet_model()
            results = model(img, verbose=False)
            st.success(f"Result: {results[0].names[results[0].probs.top1]}")

# Logic for Tab 2 (Word Image)
with tab2:
    uploaded_word_img = st.file_uploader("Word Image", type=['jpg', 'jpeg', 'png'], key="word_img")
    if uploaded_word_img:
        img = Image.open(uploaded_word_img)
        st.image(img, width=300)
        if st.button("Identify Word"):
            model = load_word_model()
            results = model.predict(source=img, save=False)
            st.success(f"Result: {results[0].names[results[0].probs.top1]}")

# Logic for Tab 3 (Word Video)
with tab3:
    st.header("🎥 Word Identification from Video")
    uploaded_video = st.file_uploader("Upload a sign language video", type=['mp4', 'mov', 'avi'], key="word_vid")
    
    if uploaded_video:
        # 1. Display the video for the user to see immediately
        st.video(uploaded_video)
        
        # 2. Prepare the video for OpenCV processing
        # We use a temporary file because cv2.VideoCapture requires a file path, not a buffer
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
        tfile.write(uploaded_video.read())
        tfile.close() # Close to ensure all data is written
        
        if st.button("🚀 Run Video Recognition"):
            cap = cv2.VideoCapture(tfile.name)
            
            # Check if video opened successfully
            if not cap.isOpened():
                st.error("Error: Could not open video file.")
            else:
                prediction_window = []
                final_word = ""
                
                # Progress UI
                status_area = st.empty()
                progress_bar = st.progress(0)
                frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                current_frame = 0

                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break
                    
                    # Perform prediction on the frame
                    model = load_word_model()
                    res = model(frame, verbose=False)
                    label = res[0].names[res[0].probs.top1]
                    
                    # Voting logic [cite: 6, 8, 9]
                    prediction_window.append(label)
                    if len(prediction_window) > 12: 
                        prediction_window.pop(0)
                    
                    if len(prediction_window) == 12:
                        common, count = Counter(prediction_window).most_common(1)[0]
                        # 8/12 threshold for stability [cite: 6, 9]
                        if count >= 8 and common not in ["Nothing", "Space"]:
                            if not final_word.endswith(common):
                                final_word += common
                    
                    # Update progress UI
                    current_frame += 1
                    if current_frame % 10 == 0: # Update UI every 10 frames to keep it smooth
                        progress_bar.progress(min(current_frame / frame_count, 1.0))
                        status_area.markdown(f"**Status:** Processing frames... Current Word: `{final_word}`")

                cap.release()
                
                # FINAL RESULT
                status_area.empty()
                progress_bar.empty()
                st.success(f"🏆 **Final Identified Word:** {final_word}")
                
            # Cleanup temp file
            if os.path.exists(tfile.name):
                os.unlink(tfile.name)
