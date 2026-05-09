import streamlit as st
import base64

st.title("🎥 Zero-Lag Native Video Player")

uploaded_video = st.file_uploader("Upload Video", type=['mp4', 'mov', 'avi'])

if uploaded_video:
    # 1. Convert video to Base64
    # This encodes the video so the HTML tag can read it as a direct data URI
    video_bytes = uploaded_video.read()
    base64_video = base64.b64encode(video_bytes).decode()
    
    # 2. Construct HTML5 Video Tag
    # We use 'autoplay' and 'controls' to ensure it plays and can be replayed
    video_html = f'''
        <video width="100%" controls autoplay>
            <source src="data:video/mp4;base64,{base64_video}" type="video/mp4">
            Your browser does not support the video tag.
        </video>
    '''
    
    # 3. Display the Video using Markdown
    # unsafe_allow_html=True is required to render the video tag
    st.markdown(video_html, unsafe_allow_html=True)
    
    st.success("Video successfully injected into browser memory for native playback.")

    # 4. Separate Analysis Trigger (For your actual app)
    if st.button("🔍 Run AI Analysis (On Disk Copy)"):
        st.info("In your real app, this button will run the YOLO loop in the background.")
