import streamlit as st
import cv2
import tempfile
import time
import os

st.title("🎥 Stable Playback Test")

uploaded_video = st.file_uploader("Upload Video", type=['mp4', 'mov', 'avi'])

if uploaded_video:
    # 1. Save file to disk
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
    tfile.write(uploaded_video.read())
    tfile.close()

    # 2. UI Setup
    # We use a container to keep the UI from shifting
    container = st.container()
    video_screen = container.empty() 
    
    if st.button("▶️ Start Playback"):
        cap = cv2.VideoCapture(tfile.name)
        
        # We manually force a slow playback (12 frames per second)
        # This ensures the browser has time to render every image.
        target_delay = 1.0 / 12  
        
        # We only process every 2nd frame to reduce data load
        frame_idx = 0
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_idx % 2 == 0:
                # 3. THE KEY FIX: Convert BGR to RGB and update the placeholder
                # We use use_container_width to keep the size moderate
                video_screen.image(frame, channels="BGR", use_container_width=True)
                
                # 4. HARD SLEEP
                # This forces the script to pause so the browser can catch up
                time.sleep(target_delay)
            
            frame_idx += 1
            
        cap.release()
        st.success("✅ Video reached the end.")

    if os.path.exists(tfile.name):
        os.unlink(tfile.name)
