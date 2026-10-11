
import os
import tempfile

import streamlit as st
from PIL import Image, ImageDraw
from inference_sdk import InferenceHTTPClient


# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="RoadGuard AI",
    page_icon="🛣️",
    layout="wide",
    initial_sidebar_state="expanded"
)

API_URL = "https://serverless.roboflow.com"
MODEL_ID = "helmet-and-no-helmet-rider-detection-8wgzg/2"


# ---------------- CUSTOM CSS ----------------
st.markdown("""
<style>
    .stApp {
        background: #E9EEF5;
        color: #233247;
    }

    .block-container {
        max-width: 1400px;
        padding-top: 0.3rem;
        padding-bottom: 2rem;
    }

    [data-testid="stHeader"] {
        background: transparent;
    }

    [data-testid="stToolbar"] {
        right: 1rem;
    }

    [data-testid="stSidebar"] {
        background: #F5F7FA;
        border-right: 1px solid #D7DFEA;
    }

    [data-testid="stSidebar"] .block-container {
        padding-top: 0.5rem;
    }

    .hero {
        background: linear-gradient(115deg, #243B5A, #426B9B);
        color: white;
        padding: 24px 28px;
        border-radius: 16px;
        margin: 0 0 20px 0;
        box-shadow: 0 5px 16px rgba(36, 59, 90, 0.14);
    }

    .hero h1 {
        color: white;
        font-size: 2.1rem;
        margin: 0 0 8px 0;
    }

    .hero p {
        color: #E6EEF8;
        font-size: 1rem;
        margin: 0;
    }

    .section-title {
        color: #243B5A;
        font-size: 1.3rem;
        font-weight: 700;
        margin: 22px 0 12px 0;
    }

    .metric-card {
        background: #FFFFFF;
        padding: 18px;
        border-radius: 12px;
        border: 1px solid #D9E2EE;
        min-height: 100px;
    }

    .metric-label {
        font-size: 0.95rem;
        color: #53657B;
        margin-bottom: 8px;
    }

    .metric-value {
        font-size: 1.8rem;
        font-weight: 750;
    }

    .stButton > button {
        background: #315D96;
        color: white;
        border: none;
        border-radius: 9px;
        padding: 0.65rem 1rem;
        font-weight: 650;
        width: 100%;
    }

    .stButton > button:hover {
        background: #243B5A;
        color: white;
    }

    div[data-testid="stFileUploader"] {
        background: white;
        border: 1px dashed #9BAEC6;
        border-radius: 12px;
        padding: 12px;
    }

    @media (max-width: 700px) {
        .hero {
            padding: 18px;
        }

        .hero h1 {
            font-size: 1.65rem;
        }
    }
</style>
""", unsafe_allow_html=True)


# ---------------- HELPER FUNCTIONS ----------------
def get_api_key():
    """Read API key from Streamlit Secrets or environment."""
    try:
        api_key = st.secrets.get("ROBOFLOW_API_KEY", "")
    except Exception:
        api_key = ""

    return api_key or os.getenv("ROBOFLOW_API_KEY", "")


def normalize_label(label):
    """Normalize labels for counting and box colors."""
    label = str(label).lower().replace("-", "_").strip()

    if "no_helmet" in label or "without_helmet" in label:
        return "no_helmet"

    if "helmet" in label:
        return "helmet"

    if (
        "license" in label
        or "number_plate" in label
        or "numberplate" in label
    ):
        return "license_plate"

    return label


def get_box_color(label):
    label = normalize_label(label)

    if label == "helmet":
        return "#16A34A"

    if label == "no_helmet":
        return "#DC2626"

    if label == "license_plate":
        return "#2563EB"

    return "#64748B"


def draw_detections(image, predictions):
    """Draw bounding boxes and confidence labels."""
    output = image.copy()
    draw = ImageDraw.Draw(output)

    image_width, image_height = output.size

    for prediction in predictions:
        try:
            x = float(prediction["x"])
            y = float(prediction["y"])
            width = float(prediction["width"])
            height = float(prediction["height"])

            x1 = max(0, int(x - width / 2))
            y1 = max(0, int(y - height / 2))
            x2 = min(image_width - 1, int(x + width / 2))
            y2 = min(image_height - 1, int(y + height / 2))

            label = str(prediction.get("class", "object"))
            confidence = float(prediction.get("confidence", 0))
            color = get_box_color(label)

            draw.rectangle(
                [x1, y1, x2, y2],
                outline=color,
                width=max(3, image_width // 300)
            )

            text = f"{label} {confidence:.0%}"
            label_y = max(0, y1 - 23)
            text_box = draw.textbbox((x1, label_y), text)

            draw.rectangle(
                [
                    text_box[0] - 3,
                    text_box[1] - 2,
                    text_box[2] + 5,
                    text_box[3] + 3
                ],
                fill=color
            )

            draw.text(
                (x1 + 2, label_y),
                text,
                fill="white"
            )

        except (KeyError, TypeError, ValueError):
            continue

    return output


def count_predictions(predictions):
    counts = {
        "helmet": 0,
        "no_helmet": 0,
        "license_plate": 0
    }

    for prediction in predictions:
        label = normalize_label(prediction.get("class", ""))

        if label in counts:
            counts[label] += 1

    return counts


def render_metric(label, value, color):
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">{label}</div>
        <div class="metric-value" style="color:{color};">
            {value}
        </div>
    </div>
    """, unsafe_allow_html=True)


# ---------------- SIDEBAR ----------------
with st.sidebar:
    with st.expander("How to Use", expanded=False):
        st.markdown("""
        **1. Upload Image**

        Choose a clear road or rider image.

        **2. Detect**

        Click the Detect Road Safety button.

        **3. View Images**

        Original Image appears on the left and
        Detected Image appears on the right.

        **4. Detection Summary**

        Check the helmet, no-helmet, and license plate counts.
        """)

    st.divider()
    st.markdown("### RoadGuard AI")
    st.caption("Intelligent Road Safety Monitoring")


# ---------------- HEADER ----------------
st.markdown("""
<div class="hero">
    <h1>RoadGuard AI</h1>
    <p>Intelligent Road Safety Monitoring using Computer Vision</p>
</div>
""", unsafe_allow_html=True)


# ---------------- IMAGE UPLOAD ----------------
st.markdown(
    '<div class="section-title">Upload Road Image</div>',
    unsafe_allow_html=True
)

uploaded_file = st.file_uploader(
    "Choose a road or rider image",
    type=["jpg", "jpeg", "png"],
    help="Upload a clear image for AI detection."
)


# ---------------- RUN DETECTION ----------------
if uploaded_file is not None:
    try:
        original_image = Image.open(uploaded_file).convert("RGB")

        detect_clicked = st.button(
            "Detect Road Safety",
            use_container_width=True
        )

        if detect_clicked:
            api_key = get_api_key()

            if not api_key:
                st.error(
                    "API key not found. Configure ROBOFLOW_API_KEY "
                    "in Streamlit Secrets or your local environment."
                )
            else:
                temp_path = None

                try:
                    with st.spinner("Detecting objects..."):
                        client = InferenceHTTPClient(
                            api_url=API_URL,
                            api_key=api_key
                        )

                        with tempfile.NamedTemporaryFile(
                            suffix=".jpg",
                            delete=False
                        ) as temp_file:
                            temp_path = temp_file.name
                            original_image.save(
                                temp_path,
                                format="JPEG"
                            )

                        result = client.infer(
                            temp_path,
                            model_id=MODEL_ID
                        )

                        predictions = result.get("predictions", [])

                        detected_image = draw_detections(
                            original_image,
                            predictions
                        )

                        counts = count_predictions(predictions)

                        st.session_state["roadguard_result"] = {
                            "original": original_image.copy(),
                            "detected": detected_image,
                            "counts": counts,
                            "uploaded_name": uploaded_file.name
                        }

                except Exception:
                    st.error(
                        "Detection failed. Check the API key, model ID, "
                        "internet connection, and Roboflow usage limits."
                    )

                finally:
                    if temp_path and os.path.exists(temp_path):
                        os.remove(temp_path)

    except Exception:
        st.error("Unable to read this image. Please upload another image.")


# ---------------- DISPLAY RESULTS ----------------
saved_result = st.session_state.get("roadguard_result")

if (
    saved_result is not None
    and uploaded_file is not None
    and saved_result.get("uploaded_name") == uploaded_file.name
):
    # Original image on the left; detected image on the right.
    left_col, right_col = st.columns(2, gap="large")

    with left_col:
        st.markdown("**Original Image**")
        st.image(
            saved_result["original"],
            use_container_width=True
        )

    with right_col:
        st.markdown("**Detected Image**")
        st.image(
            saved_result["detected"],
            use_container_width=True
        )

    # Detection summary only.
    st.markdown(
        '<div class="section-title">Detection Summary</div>',
        unsafe_allow_html=True
    )

    counts = saved_result["counts"]

    col1, col2, col3 = st.columns(3, gap="medium")

    with col1:
        render_metric(
            "🟢 Helmet",
            counts["helmet"],
            "#16A34A"
        )

    with col2:
        render_metric(
            "🔴 No Helmet",
            counts["no_helmet"],
            "#DC2626"
        )

    with col3:
        render_metric(
            "🔵 License Plate",
            counts["license_plate"],
            "#2563EB"
        )
