import streamlit as st
import cv2
import tempfile
import time
import os

st.title("🎥 Codec-Independent Player")

uploaded_video = st.file_uploader("Upload Video", type=['mp4', 'mov', 'avi'])

if uploaded_video:
    # 1. Save to disk
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
    tfile.write(uploaded_video.read())
    tfile.close()

    # 2. Setup Screen
    video_screen = st.empty()
    cap = cv2.VideoCapture(tfile.name)
    
    # Get Metadata
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0: fps = 30
    frame_delay = 1.0 / fps

    if st.button("▶️ Play Video"):
        frame_count = 0
        while cap.isOpened():
            start_time = time.time()
            ret, frame = cap.read()
            if not ret: break
            
            # --- THE STABILITY FIX ---
            # We only update the UI every 2nd frame. 
            # This cuts the data load in half and prevents freezing.
            if frame_count % 2 == 0:
                video_screen.image(frame, channels="BGR", use_container_width=True)
            
            # --- THE DURATION FIX ---
            # Maintain the actual video speed
            elapsed = time.time() - start_time
            time.sleep(max(0, frame_delay - elapsed))
            
            frame_count += 1
            
        cap.release()
        st.success("Playback Finished.")

    if os.path.exists(tfile.name):
        os.unlink(tfile.name)
