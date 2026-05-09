import streamlit as st
import cv2
import tempfile
import time
import os

st.set_page_config(page_title="Video Playback Test", layout="centered")
st.title("🎥 Native Duration Video Player")

# 1. File Uploader
uploaded_video = st.file_uploader("Upload a video to test playback", type=['mp4', 'mov', 'avi'])

if uploaded_video:
    # 2. Save to a temporary file
    # This is necessary because OpenCV needs a file path to read metadata like FPS
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
    tfile.write(uploaded_video.read())
    tfile.close()

    # 3. Initialize the Screen and Metadata
    video_screen = st.empty()
    cap = cv2.VideoCapture(tfile.name)
    
    # Get original Video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Validation: Fallback if metadata is missing
    if fps <= 0: fps = 30 
    
    # Calculate how long to wait between frames to match real duration
    frame_delay = 1.0 / fps 
    duration = total_frames / fps

    st.info(f"Video Info: {total_frames} frames | {fps} FPS | Duration: {duration:.2f} seconds")

    # 4. Playback Button
    if st.button("▶️ Play Video"):
        current_frame = 0
        
        while cap.isOpened():
            start_time = time.time() # Start timing this frame
            
            ret, frame = cap.read()
            if not ret:
                break
            
            # Display frame (Convert BGR to RGB for Streamlit)
            video_screen.image(frame, channels="BGR", use_container_width=True)
            
            # --- THE DURATION FIX ---
            # Calculate how long it took to render and wait the remainder
            # This ensures the video plays at its actual speed
            elapsed = time.time() - start_time
            time.sleep(max(0, frame_delay - elapsed))
            
            current_frame += 1

        cap.release()
        st.success("Playback Finished.")

    # 5. Cleanup
    if os.path.exists(tfile.name):
        os.unlink(tfile.name)
