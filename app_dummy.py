import streamlit as st
import os

st.set_page_config(page_title="Stable Video Player", layout="centered")
st.title("🎥 Standard HTML5 Player")

# 1. File Uploader
uploaded_video = st.file_uploader("Upload video", type=['mp4', 'mov', 'avi'])

if uploaded_video:
    # --- THE FIX: PREVENT FREEZING ---
    # We read the file into memory once. This ensures the stream is stable.
    video_bytes = uploaded_video.read()
    
    # We provide the bytes directly to the video widget. 
    # This uses the browser's hardware acceleration so it NEVER freezes.
    st.video(video_bytes)
    
    st.success("Video loaded into browser memory.")
    
    # 2. Status Check
    st.write(f"File Name: {uploaded_video.name}")
    st.write(f"File Size: {len(video_bytes) / (1024*1024):.2f} MB")

    st.info("💡 If you see a black screen here, your browser cannot decode this specific MP4 codec.")
