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
import time

with tab3:
    st.header("🎥 Word Identification")
    uploaded_video = st.file_uploader("Upload a sign video", type=['mp4', 'mov', 'avi'], key="w_v")
    
    if uploaded_video:
        # --- 1. RESET POINTERS ---
        uploaded_video.seek(0)
        
        # Save to temp file for OpenCV
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
        tfile.write(uploaded_video.read())
        tfile.close() 
        
        # --- 2. THE PLAYER (No Thumbnail) ---
        # We reset the pointer again so the video widget doesn't stay 'black'
        uploaded_video.seek(0)
        
        # Use a unique key to prevent Streamlit from 'sticking' to old thumbnails
        st.video(uploaded_video, autoplay=True, loop=True, key=f"video_{len(uploaded_video.name)}")
        
        st.divider()

        # --- 3. ANALYSIS SECTION ---
        # Create a container that is ONLY populated when the button is clicked
        analysis_container = st.container()
        
        if st.button("🔍 Start AI Word Analysis"):
            with analysis_container:
                cap = cv2.VideoCapture(tfile.name)
                
                if not cap.isOpened():
                    st.error("Error: Video format not supported by OpenCV.")
                else:
                    # Moderate sized columns for the AI output
                    col1, col2, col3 = st.columns([1, 2, 1])
                    with col2:
                        # Use st.empty() so the frame replaces itself instantly (No thumbnail)
                        video_screen = st.empty() 
                        # We use a simple text status instead of a progress bar to keep it clean
                        status_text = st.empty()

                    # Logic Settings
                    WINDOW_SIZE, VOTE_THRESHOLD = 12, 8
                    final_word, last_word, prediction_window = "", None, []
                    
                    model = load_word_model()

                    while cap.isOpened():
                        ret, frame = cap.read()
                        if not ret: break 
                        
                        # AI Inference
                        results = model(frame, verbose=False)
                        label = results[0].names[results[0].probs.top1]
                        
                        # Voting Logic
                        prediction_window.append(label)
                        if len(prediction_window) > WINDOW_SIZE: prediction_window.pop(0)
                        if len(prediction_window) == WINDOW_SIZE:
                            common, count = Counter(prediction_window).most_common(1)[0]
                            if count >= VOTE_THRESHOLD and common != last_word:
                                if common not in ["Nothing", "Space"]:
                                    final_word += f" {common}"
                                    last_word = common
                                    prediction_window = []

                        # UPDATE THE SCREEN: No progress bar, no thumbnail
                        video_screen.image(frame, channels="BGR", use_container_width=True)
                        status_text.markdown(f"### 🔤 Detected: `{final_word}`")
                        
                        # Controlled Slow Playback
                        time.sleep(0.04) 

                    cap.release()
                    st.success(f"🏆 Final Word: **{final_word}**")

        # Cleanup temp file
        if os.path.exists(tfile.name):
            try: os.unlink(tfile.name)
            except: pass
