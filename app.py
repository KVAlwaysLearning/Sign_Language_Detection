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
import time # Ensure this is at the top of app.py

with tab3:
    st.header("🎥 Word Identification (Auto-Play & Analyze)")
    uploaded_video = st.file_uploader("Upload a sign video", type=['mp4', 'mov', 'avi'], key="w_v")
    
    if uploaded_video:
        final_word = "" 
        
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
        tfile.write(uploaded_video.read())
        tfile.close() 

        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            video_screen = st.empty()
            progress_bar = st.progress(0)
            status_text = st.empty()
        
        if st.button("▶️ Start Recognition & Playback"):
            cap = cv2.VideoCapture(tfile.name)
            
            if not cap.isOpened():
                st.error("Cannot play video.")
            else:
                # Get the actual frame rate of the video to match speed
                fps = cap.get(cv2.CAP_PROP_FPS)
                if fps == 0 or fps > 60: fps = 30 # Default fallback
                frame_delay = 1.0 / fps # Time to wait between frames

                WINDOW_SIZE, VOTE_THRESHOLD = 12, 8
                last_word, prediction_window = None, []
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                current_frame = 0
                
                model = load_word_model()

                while cap.isOpened():
                    start_time = time.time() # Record start of frame processing
                    
                    ret, frame = cap.read()
                    if not ret: break 
                    
                    # 1. AI Analysis
                    results = model(frame, verbose=False)
                    label = results[0].names[results[0].probs.top1]
                    
                    # 2. Voting Logic
                    prediction_window.append(label)
                    if len(prediction_window) > WINDOW_SIZE: prediction_window.pop(0)
                    if len(prediction_window) == WINDOW_SIZE:
                        common, count = Counter(prediction_window).most_common(1)[0]
                        if count >= VOTE_THRESHOLD and common != last_word:
                            if common not in ["Nothing", "Space"]:
                                final_word += f" {common}"
                                last_word = common
                                prediction_window = []

                    # 3. Update Video Screen
                    video_screen.image(frame, channels="BGR", use_container_width=True)
                    
                    current_frame += 1
                    progress_bar.progress(min(current_frame / total_frames, 1.0))
                    status_text.markdown(f"**Detecting:** `{final_word}`")

                    # --- THE FIX: CONTROLLED DELAY ---
                    # Calculate how long the AI took and wait the remaining time
                    processing_time = time.time() - start_time
                    wait_time = max(0, frame_delay - processing_time)
                    time.sleep(wait_time) # Force the loop to match video speed

                cap.release()
                st.success(f"🏆 Final Identified Word: **{final_word}**")

        if os.path.exists(tfile.name):
            try: os.unlink(tfile.name)
            except: pass
