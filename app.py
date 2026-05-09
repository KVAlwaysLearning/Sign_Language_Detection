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
    st.header("🎥 Word Identification (Video)")
    uploaded_video = st.file_uploader("Upload a sign video", type=['mp4', 'mov', 'avi'], key="w_v")
    
    if uploaded_video:
        # --- THE FIX: RESET BUFFER ---
        # Reset pointer to start so we can write the temp file
        uploaded_video.seek(0)
        
        # Save to a temporary file for OpenCV (cv2 requires a file path) 
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
        tfile.write(uploaded_video.read())
        tfile.close() 

        # Reset pointer AGAIN so st.video can display it in the browser
        uploaded_video.seek(0)
        st.video(uploaded_video)
        
        if st.button("Analyze Video"):
            # Load the word model
            model = load_word_model()
            cap = cv2.VideoCapture(tfile.name)
            
            if not cap.isOpened():
                st.error("OpenCV could not open this video format. Try a standard H.264 MP4.")
            else:
                # --- PREDICTION SETTINGS ---
                WINDOW_SIZE = 12       # Look at 12 frames at a time 
                VOTE_THRESHOLD = 8     # Must see the same word 8/12 times 
                
                final_word = ""
                last_committed_word = None
                prediction_window = []
                
                # UI Placeholders for live updates
                status_text = st.empty()
                progress_bar = st.progress(0)
                frame_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                current_frame = 0

                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break
                    
                    # 1. Predict current frame [cite: 7]
                    results = model(frame, verbose=False)
                    label = results[0].names[results[0].probs.top1]
                    
                    # 2. Add to rolling window [cite: 8]
                    prediction_window.append(label)
                    if len(prediction_window) > WINDOW_SIZE:
                        prediction_window.pop(0)
                    
                    # 3. Analyze window for a "Stable" word [cite: 9]
                    if len(prediction_window) == WINDOW_SIZE:
                        counts = Counter(prediction_window)
                        most_common, count = counts.most_common(1)[0]
                        
                        # Only commit if stable and not the same as the previous word [cite: 9]
                        if count >= VOTE_THRESHOLD and most_common != last_committed_word:
                            if most_common not in ["Nothing", "Space"]:
                                final_word += f" {most_common}"
                                last_committed_word = most_common
                                # Optional: Clear window to wait for next distinct sign [cite: 10]
                                prediction_window = [] 
                            
                            elif most_common == "Space":
                                final_word += " "
                                last_committed_word = most_common
                                prediction_window = []

                    # Update Progress
                    current_frame += 1
                    if current_frame % 5 == 0:
                        progress_bar.progress(min(current_frame / frame_total, 1.0))
                        status_text.write(f"Analyzing... Current Word(s): **{final_word}**")

                cap.release()
                
                # --- FINAL RESULT ---
                status_text.empty()
                progress_bar.empty()
                st.success(f"🏆 Final Identified Word(s): \b{final_word}\b")
                
            # Cleanup the temporary file from the server
            if os.path.exists(tfile.name):
                os.unlink(tfile.name)
