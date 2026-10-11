
import os

import streamlit as st
from PIL import Image, ImageDraw, ImageFont
from inference_sdk import InferenceHTTPClient


# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------
st.set_page_config(
    page_title="RoadGuard AI",
    page_icon="🛣️",
    layout="wide",
    initial_sidebar_state="expanded"
)


# --------------------------------------------------
# THEME / CSS
# --------------------------------------------------
st.markdown("""
<style>
    .stApp {
        background: #E9EEF5;
    }

    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 2rem;
        max-width: 1400px;
    }

    .hero {
        background: linear-gradient(120deg, #243B5A, #426B9B);
        padding: 26px;
        border-radius: 16px;
        text-align: center;
        margin-bottom: 24px;
        box-shadow: 0 5px 15px rgba(36, 59, 90, 0.16);
    }

    .hero h1 {
        color: white;
        margin: 0;
        font-size: 36px;
    }

    .hero p {
        color: #E1ECF8;
        margin: 8px 0 0;
        font-size: 16px;
    }

    .section-title {
        color: #315D96;
        border-left: 5px solid #426B9B;
        padding-left: 12px;
        margin: 20px 0 16px;
    }

    .result-card {
        background: #FFFFFF;
        border: 1px solid #D4DFEC;
        border-radius: 12px;
        padding: 18px;
        text-align: center;
        box-shadow: 0 3px 10px rgba(36, 59, 90, 0.07);
    }

    .result-number {
        font-size: 30px;
        font-weight: 700;
        margin: 5px 0;
    }

    .result-label {
        color: #475569;
        font-size: 14px;
        font-weight: 600;
    }

    .stButton > button {
        background: #315D96;
        color: white;
        border: none;
        border-radius: 9px;
        padding: 10px 18px;
        font-weight: 600;
    }

    .stButton > button:hover {
        background: #243B5A;
        color: white;
        border: none;
    }
</style>
""", unsafe_allow_html=True)


# --------------------------------------------------
# HEADER
# --------------------------------------------------
st.markdown("""
<div class="hero">
    <h1>🛣️ RoadGuard AI</h1>
    <p>Intelligent Road Safety Monitoring System</p>
    <p>AI-Powered Helmet Detection • Road Safety • Smart Monitoring</p>
</div>
""", unsafe_allow_html=True)


# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------
with st.sidebar:
    st.markdown("## 🛣️ RoadGuard AI")
    st.caption("Road safety through AI")

    st.markdown("---")
    st.markdown("### 📌 Detection Legend")
    st.markdown("🟢 **Helmet**")
    st.markdown("🔴 **No Helmet**")
    st.markdown("🔵 **License Plate**")

    st.markdown("---")
    st.caption("Upload a road image to analyze it.")


# --------------------------------------------------
# ROBoflow CONFIGURATION
# --------------------------------------------------
API_URL = "https://serverless.roboflow.com"
MODEL_ID = "helmet-and-no-helmet-rider-detection-8wgzg/2"


def get_api_key():
    """Read the API key from Streamlit Secrets or environment."""
    try:
        secret_key = st.secrets.get("ROBOFLOW_API_KEY", "")
    except Exception:
        secret_key = ""

    return secret_key or os.getenv("ROBOFLOW_API_KEY", "")


# --------------------------------------------------
# HELPER FUNCTIONS
# --------------------------------------------------
def normalize_label(label):
    return str(label).lower().strip().replace(" ", "_").replace("-", "_")


def get_box_color(label):
    label = normalize_label(label)

    if label == "helmet":
        return "#22C55E"

    if label == "no_helmet":
        return "#EF4444"

    if label == "license_plate":
        return "#3B82F6"

    return "#F59E0B"


def get_font(size=18):
    try:
        return ImageFont.truetype("DejaVuSans.ttf", size)
    except OSError:
        return ImageFont.load_default()


def draw_detections(original_image, predictions):
    """Draw colored bounding boxes and labels on the uploaded image."""
    result = original_image.copy().convert("RGB")
    draw = ImageDraw.Draw(result)
    font = get_font(18)

    image_width, image_height = result.size

    for prediction in predictions:
        label = prediction.get("class", "object")
        confidence = float(prediction.get("confidence", 0))

        x = float(prediction.get("x", 0))
        y = float(prediction.get("y", 0))
        box_width = float(prediction.get("width", 0))
        box_height = float(prediction.get("height", 0))

        # Roboflow provides center coordinates and box dimensions
        x1 = max(0, int(x - box_width / 2))
        y1 = max(0, int(y - box_height / 2))
        x2 = min(image_width - 1, int(x + box_width / 2))
        y2 = min(image_height - 1, int(y + box_height / 2))

        color = get_box_color(label)

        draw.rectangle(
            [x1, y1, x2, y2],
            outline=color,
            width=4
        )

        display_label = f"{label.replace('_', ' ').title()} {confidence:.0%}"

        text_box = draw.textbbox((x1, y1), display_label, font=font)
        text_height = text_box[3] - text_box[1]
        text_width = text_box[2] - text_box[0]

        label_y = max(0, y1 - text_height - 10)

        draw.rectangle(
            [
                x1,
                label_y,
                min(image_width - 1, x1 + text_width + 10),
                min(image_height - 1, label_y + text_height + 8)
            ],
            fill=color
        )

        draw.text(
            (x1 + 5, label_y + 3),
            display_label,
            fill="white",
            font=font
        )

    return result


def render_count_card(label, count, color, background):
    st.markdown(f"""
    <div class="result-card">
        <div style="font-size: 27px;">{label}</div>
        <div class="result-number" style="color: {color};">
            {count}
        </div>
        <div class="result-label">Detections</div>
    </div>
    """, unsafe_allow_html=True)


# --------------------------------------------------
# IMAGE UPLOAD
# --------------------------------------------------
st.markdown(
    '<h2 class="section-title">📤 Upload Road Image</h2>',
    unsafe_allow_html=True
)

uploaded_file = st.file_uploader(
    "Choose a road image",
    type=["jpg", "jpeg", "png"],
    help="Upload a clear road image for helmet detection."
)


# --------------------------------------------------
# PROCESS IMAGE
# --------------------------------------------------
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")

    st.success("Image uploaded successfully!")

    api_key = get_api_key()

    if not api_key:
        st.error(
            "Roboflow API key is not configured. "
            "Add ROBOFLOW_API_KEY to Streamlit Secrets."
        )
        st.stop()

    if st.button("🔍 Detect Road Safety", use_container_width=True):
        with st.spinner("Analyzing image with RoadGuard AI..."):
            try:
                client = InferenceHTTPClient(
                    api_url=API_URL,
                    api_key=api_key
                )

                response = client.infer(
                    image,
                    model_id=MODEL_ID
                )

                predictions = response.get("predictions", [])
                result_image = draw_detections(image, predictions)

                # Count supported classes
                helmet_count = 0
                no_helmet_count = 0
                license_plate_count = 0

                for prediction in predictions:
                    label = normalize_label(
                        prediction.get("class", "")
                    )

                    if label == "helmet":
                        helmet_count += 1
                    elif label == "no_helmet":
                        no_helmet_count += 1
                    elif label == "license_plate":
                        license_plate_count += 1

                # Store results for display
                st.session_state["roadguard_original"] = image
                st.session_state["roadguard_result"] = result_image
                st.session_state["roadguard_predictions"] = predictions
                st.session_state["roadguard_counts"] = (
                    helmet_count,
                    no_helmet_count,
                    license_plate_count
                )

            except Exception as error:
                st.error(
                    "Detection failed. Check your Roboflow model, "
                    "API key, internet connection, and API usage."
                )
                st.caption(f"Technical details: {error}")


# --------------------------------------------------
# DISPLAY IMAGES SIDE BY SIDE
# --------------------------------------------------
if "roadguard_result" in st.session_state:
    original = st.session_state["roadguard_original"]
    result = st.session_state["roadguard_result"]

    st.markdown(
        '<h2 class="section-title">🖼️ Image Comparison</h2>',
        unsafe_allow_html=True
    )

    left_col, right_col = st.columns(2, gap="large")

    with left_col:
        st.markdown("### 📷 Original Image")
        st.image(
            original,
            caption="Uploaded Road Image",
            use_container_width=True
        )

    with right_col:
        st.markdown("### 🔍 Detection Image")
        st.image(
            result,
            caption="RoadGuard AI Detection Result",
            use_container_width=True
        )

    # --------------------------------------------------
    # DETECTION RESULTS - FULL WIDTH, LEFT ALIGNED
    # --------------------------------------------------
    st.markdown("---")
    st.markdown(
        '<h2 class="section-title">🎯 Detection Result</h2>',
        unsafe_allow_html=True
    )

    helmet_count, no_helmet_count, license_plate_count = (
        st.session_state["roadguard_counts"]
    )

    col1, col2, col3 = st.columns(3, gap="medium")

    with col1:
        render_count_card(
            "🟢",
            helmet_count,
            "#21834A",
            "#E2F4E8"
        )

    with col2:
        render_count_card(
            "🔴",
            no_helmet_count,
            "#C83D45",
            "#FBE5E5"
        )

    with col3:
        render_count_card(
            "🔵",
            license_plate_count,
            "#315D96",
            "#E1ECFC"
        )

    # Safety summary
    st.markdown("### 🛡️ Road Safety Summary")

    if no_helmet_count > 0:
        st.warning(
            f"⚠️ {no_helmet_count} no-helmet detection(s) found. "
            "Please review the marked areas."
        )
    elif helmet_count > 0:
        st.success(
            "✅ Helmet detection(s) found and no no-helmet "
            "detections were returned by the model."
        )
    else:
        st.info(
            "No helmet or no-helmet detections were returned. "
            "Try a clearer image if necessary."
        )

    st.caption(
        "AI results may contain errors. Please verify detections "
        "before using them for real-world decisions."
    )

    # Optional prediction details
    with st.expander("📋 View Detection Details"):
        predictions = st.session_state["roadguard_predictions"]

        if predictions:
            for index, prediction in enumerate(predictions, start=1):
                label = prediction.get("class", "Unknown")
                confidence = float(
                    prediction.get("confidence", 0)
                )

                st.write(
                    f"{index}. **{label.replace('_', ' ').title()}** "
                    f"— Confidence: {confidence:.1%}"
                )
        else:
            st.write("No objects were detected in this image.")

else:
    st.info(
        "Upload an image and click **Detect Road Safety** "
        "to view the original image, detection image, and results."
    )


# --------------------------------------------------
# FOOTER
# --------------------------------------------------
st.markdown("---")
st.markdown(
    """
    <div style="text-align:center; color:#64748B; padding:10px;">
        🛣️ <b>RoadGuard AI</b> · Intelligent Road Safety Monitoring
    </div>
    """,
    unsafe_allow_html=True
)
