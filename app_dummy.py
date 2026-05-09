import streamlit as st
import cv2
import tempfile
import time
import os

st.title("🎥 Controlled Duration Player")

uploaded_video = st.file_uploader("Upload Video", type=['mp4', 'mov', 'avi'])

if uploaded_video:
    # 1. Save to disk
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
    tfile.write(uploaded_video.read())
    tfile.close()

    # 2. UI Setup
    video_screen = st.empty()
    status_text = st.empty()
    
    cap = cv2.VideoCapture(tfile.name)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # --- THE PLAYTIME FIX ---
    # If the video feels too fast, we manually set a slow FPS (e.g., 15 or 20)
    # This forces the video to stay on screen longer.
    target_fps = 18  # Lower this number to make it even slower
    frame_delay = 1.0 / target_fps 

    if st.button("▶️ Start Slow Playback"):
        current_frame = 0
        
        while cap.isOpened():
            start_time = time.time()
            ret, frame = cap.read()
            
            if not ret:
                break
            
            # Update the UI
            video_screen.image(frame, channels="BGR", use_container_width=True)
            
            # Progress feedback
            current_frame += 1
            status_text.text(f"Playing frame {current_frame} of {total_frames}...")

            # --- FORCED WAIT ---
            # This is the "Anti-Lag" logic. It ensures the frame stays 
            # visible for the exact 'frame_delay' duration.
            elapsed = time.time() - start_time
            sleep_time = max(0, frame_delay - elapsed)
            time.sleep(sleep_time)
            
        cap.release()
        status_text.success("✅ Playback Complete")

    if os.path.exists(tfile.name):
        os.unlink(tfile.name)
