"""
app.py - Streamlit web app for AQI prediction

Run with:
    streamlit run app.py

Features:
- Upload a sky/pollution image
- Optional text description input
- Displays predicted AQI with color-coded category
- Shows AQI scale reference
"""

import os
import sys
import streamlit as st
from PIL import Image

sys.path.insert(0, os.path.dirname(__file__))

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AQI Predictor",
    page_icon="🌫️",
    layout="centered",
)

# ── AQI color scale ───────────────────────────────────────────────────────────
AQI_SCALE = [
    (0,   50,  "Good",                          "#00e400"),
    (51,  100, "Moderate",                      "#ffff00"),
    (101, 150, "Unhealthy for Sensitive Groups","#ff7e00"),
    (151, 200, "Unhealthy",                     "#ff0000"),
    (201, 300, "Very Unhealthy",                "#8f3f97"),
    (301, 500, "Hazardous",                     "#7e0023"),
]

def get_color(aqi):
    for lo, hi, _, color in AQI_SCALE:
        if lo <= aqi <= hi:
            return color
    return "#7e0023"

def get_category(aqi):
    for lo, hi, label, _ in AQI_SCALE:
        if lo <= aqi <= hi:
            return label
    return "Hazardous"


# ── UI ────────────────────────────────────────────────────────────────────────
st.title("🌫️ Air Quality Index Predictor")
st.markdown("Upload a sky or pollution image to predict the AQI using a Vision-Language Model.")

# Check if model exists
CHECKPOINT = os.path.join(os.path.dirname(__file__), "checkpoints", "best_model.pt")
if not os.path.exists(CHECKPOINT):
    st.error("No trained model found. Run `python src/train.py` first to train the model.")
    st.stop()

# Image upload
uploaded_file = st.file_uploader("Upload sky/pollution image", type=["jpg", "jpeg", "png"])

# Optional text input
text_input = st.text_input(
    "Optional: Add weather/location context",
    placeholder="e.g. Hazy morning in Delhi, 32°C, humidity 80%",
)

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    if st.button("Predict AQI", type="primary"):
        with st.spinner("Analyzing image..."):
            try:
                # Save temp file for inference
                import tempfile
                with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
                    image.save(tmp.name)
                    tmp_path = tmp.name

                from src.inference import predict
                result = predict(
                    image_path=tmp_path,
                    text=text_input if text_input.strip() else None,
                    checkpoint_path=CHECKPOINT,
                )
                os.unlink(tmp_path)

                aqi      = result["aqi"]
                category = result["category"]
                color    = get_color(aqi)

                # Display result
                st.markdown("---")
                col1, col2 = st.columns(2)

                with col1:
                    st.metric("Predicted AQI", f"{aqi}")

                with col2:
                    st.markdown(
                        f"<div style='background:{color};padding:12px;border-radius:8px;"
                        f"text-align:center;font-weight:bold;color:{'black' if aqi < 101 else 'white'}'>"
                        f"{category}</div>",
                        unsafe_allow_html=True,
                    )

                # AQI progress bar
                st.markdown("**AQI Scale:**")
                st.progress(min(aqi / 500.0, 1.0))

            except Exception as e:
                st.error(f"Prediction failed: {e}")

# AQI reference table
with st.expander("AQI Reference Scale"):
    for lo, hi, label, color in AQI_SCALE:
        st.markdown(
            f"<span style='background:{color};padding:3px 8px;border-radius:4px;"
            f"color:{'black' if lo < 101 else 'white'}'>{lo}-{hi}</span> &nbsp; {label}",
            unsafe_allow_html=True,
        )
        st.write("")
