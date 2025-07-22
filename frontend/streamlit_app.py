import streamlit as st
import requests
from PIL import Image
import io
import base64

API_URL = "http://backend:8000/predict"

st.set_page_config(page_title="YOLOv8 Object Detection", layout="centered")
st.title("YOLOv8 - Technical Drawing Symbol Detector")

uploaded_file = st.file_uploader("Upload an image (JPG or PNG)", type=["jpg", "jpeg", "png"])

# Run detection and store result in session_state
if uploaded_file and st.button("Run Detection"):
    with st.spinner("Detecting..."):
        files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
        try:
            response = requests.post(API_URL, files=files)
            response.raise_for_status()
            data = response.json()

            # Save results in session_state
            st.session_state["processed_image_bytes"] = base64.b64decode(data["processed_image"])
            st.session_state["detection_results"] = data["detections"]
            st.session_state["original_filename"] = uploaded_file.name

        except requests.exceptions.RequestException as e:
            st.error(f"Request failed: {e}")
        except Exception as e:
            st.error(f"Error: {e}")

# Display stored results if available
if "processed_image_bytes" in st.session_state and "detection_results" in st.session_state:
    st.subheader("Processed Output:")

    # Show processed image
    image = Image.open(io.BytesIO(st.session_state["processed_image_bytes"]))
    st.image(image, caption="Processed Image", use_container_width=True)

    # Download button for image
    st.download_button(
        label="📥 Download Processed Image",
        data=st.session_state["processed_image_bytes"],
        file_name="processed_image.jpg",
        mime="image/jpeg"
    )

    # Show detection info
    st.subheader("Detected Objects:")
    detection_text = ""
    for i, det in enumerate(st.session_state["detection_results"]):
        info = {
            "Object #": i + 1,
            "Class ID": det["class_id"],
            "Confidence": round(det["confidence"], 3),
            "BBox (x1, y1, x2, y2)": det["bbox"]
        }
        st.json(info)
        detection_text += f"{info}\n"

    # Download button for detection info
    st.download_button(
        label="📄 Download Detection Info",
        data=detection_text,
        file_name="detection_info.txt",
        mime="text/plain"
    )
