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
import streamlit as st
import cv2
import tempfile
import os
from collections import Counter

with tab3:
    st.header("🎥 Word Identification (Background Analysis)")
    uploaded_video = st.file_uploader("Upload a sign video for analysis", type=['mp4', 'mov', 'avi'], key="w_v")
    
    if uploaded_video:
        # 1. Save to temporary file
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
        tfile.write(uploaded_video.read())
        tfile.close() 

        st.info("✅ Video uploaded. Click the button below to start the background AI analysis.")

        # 2. ANALYSIS BUTTON
        if st.button("🔍 Start AI Word Analysis"):
            cap = cv2.VideoCapture(tfile.name)
            
            if not cap.isOpened():
                st.error("Error: System could not read the video file.")
            else:
                # --- Analysis Setup ---
                WINDOW_SIZE, VOTE_THRESHOLD = 12, 8
                final_word, last_word, prediction_window = "", None, []
                
                # Metadata for progress tracking
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                current_frame = 0
                
                # UI Placeholders
                progress_bar = st.progress(0)
                status_text = st.empty()

                # --- BACKGROUND LOOP ---
                # We do NOT use video_screen.image() here to prevent lag
                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break
                    
                    # AI Inference
                    word_model=load_word_model()
                    results = word_model(frame, verbose=False)
                    label = results[0].names[results[0].probs.top1]
                    
                    # Voting Logic
                    prediction_window.append(label)
                    if len(prediction_window) > WINDOW_SIZE: 
                        prediction_window.pop(0)
                    
                    if len(prediction_window) == WINDOW_SIZE:
                        counts = Counter(prediction_window)
                        common, count = counts.most_common(1)[0]
                        
                        if count >= VOTE_THRESHOLD and common != last_word:
                            if common not in ["Nothing", "Space"]:
                                final_word += f" {common}"
                                last_word = common
                                prediction_window = [] # Reset to detect next word

                    # Update progress bar every 5 frames
                    current_frame += 1
                    if current_frame % 5 == 0:
                        progress_val = min(current_frame / total_frames, 1.0)
                        progress_bar.progress(progress_val)
                        status_text.text(f"Processing frame {current_frame}/{total_frames}...")

                cap.release()
                
                # --- 3. FINAL OUTPUT ---
                progress_bar.empty()
                status_text.empty()
                
                if final_word.strip():
                    st.success(f"🏆 **Final Identified Word(s):** {final_word.strip()}")
                else:
                    st.warning("⚠️ No sign language words were clearly detected in this video.")

        # Cleanup temp file
        if os.path.exists(tfile.name):
            try:
                os.unlink(tfile.name)
            except:
                pass
