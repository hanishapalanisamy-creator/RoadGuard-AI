
import os
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
        padding-top: 1rem;
    }

    .hero {
        background: linear-gradient(115deg, #243B5A, #426B9B);
        color: white;
        padding: 25px 28px;
        border-radius: 16px;
        margin: 0 0 22px 0;
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
        font-size: 1.35rem;
        font-weight: 700;
        margin: 20px 0 12px 0;
    }

    .info-card {
        background: white;
        padding: 17px;
        border-radius: 12px;
        border: 1px solid #D9E2EE;
        margin-bottom: 12px;
    }

    .info-card h4 {
        color: #243B5A;
        margin: 0 0 8px 0;
    }

    .info-card p {
        color: #53657B;
        margin: 0;
        font-size: 0.95rem;
    }

    .metric-card {
        background: white;
        padding: 18px;
        border-radius: 12px;
        border: 1px solid #D9E2EE;
        text-align: center;
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

    .status-box {
        background: #FFFFFF;
        border-left: 5px solid #315D96;
        border-radius: 10px;
        padding: 16px;
        margin: 12px 0;
        color: #34465D;
    }

    .footer {
        text-align: center;
        color: #66788E;
        font-size: 0.85rem;
        padding-top: 28px;
    }

    @media (max-width: 700px) {
        .hero {
            padding: 20px;
        }

        .hero h1 {
            font-size: 1.65rem;
        }
    }
</style>
""", unsafe_allow_html=True)


# ---------------- HEADER ----------------
st.markdown("""
<div class="hero">
    <h1>🛣️ RoadGuard AI</h1>
    <p>Intelligent Road Safety Monitoring using Computer Vision</p>
</div>
""", unsafe_allow_html=True)


# ---------------- SIDEBAR ----------------
with st.sidebar:
    st.markdown("## 🛡️ RoadGuard AI")
    st.caption("AI-powered road safety detection")

    st.divider()

    with st.expander("📘 How to Use", expanded=False):
        st.markdown("""
        **1. Upload Image**  
        Choose a clear road or rider image.

        **2. Run Detection**  
        Click the detection button.

        **3. Compare Images**  
        View the original and detected images.

        **4. Check Results**  
        Review detected object counts below.

        ---
        **💡 Tips**
        - Use a clear, well-lit image.
        - Keep the rider visible.
        - Results depend on image quality and model accuracy.
        """)

    with st.expander("🎨 Detection Legend", expanded=False):
        st.markdown("""
        🟢 **Green:** Helmet detected  
        
        🔴 **Red:** No helmet detected  
        
        🔵 **Blue:** License plate detected
        """)

    st.divider()
    st.caption("Computer Vision Project")


# ---------------- HELPER FUNCTIONS ----------------
def get_api_key():
    """Read the API key from Streamlit Secrets or environment variables."""
    try:
        key = st.secrets.get("ROBOFLOW_API_KEY", "")
    except Exception:
        key = ""

    if not key:
        key = os.getenv("ROBOFLOW_API_KEY", "")

    return key


def normalize_label(label):
    """Normalize model labels for consistent counting and coloring."""
    label = str(label).lower().replace("-", "_").strip()

    if "no_helmet" in label or "without_helmet" in label:
        return "no_helmet"
    if "helmet" in label:
        return "helmet"
    if "license" in label or "number_plate" in label or "numberplate" in label:
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
    """Draw detection boxes using Roboflow center-based coordinates."""
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

            # Draw a readable label background.
            text_box = draw.textbbox((x1, max(0, y1 - 22)), text)
            label_y = max(0, y1 - 22)

            draw.rectangle(
                [
                    text_box[0] - 3,
                    label_y - 2,
                    text_box[2] + 5,
                    text_box[3] + 3
                ],
                fill=color
            )

            draw.text((x1 + 2, label_y), text, fill="white")

        except (KeyError, TypeError, ValueError):
            continue

    return output


def count_predictions(predictions):
    counts = {
        "helmet": 0,
        "no_helmet": 0,
        "license_plate": 0,
        "other": 0
    }

    for prediction in predictions:
        label = normalize_label(prediction.get("class", ""))

        if label in ("helmet", "no_helmet", "license_plate"):
            counts[label] += 1
        else:
            counts["other"] += 1

    return counts


def render_metric(label, value, color):
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">{label}</div>
        <div class="metric-value" style="color:{color};">{value}</div>
    </div>
    """, unsafe_allow_html=True)


# ---------------- IMAGE UPLOAD ----------------
st.markdown(
    '<div class="section-title">📤 Upload Road Image</div>',
    unsafe_allow_html=True
)

uploaded_file = st.file_uploader(
    "Choose a road or rider image",
    type=["jpg", "jpeg", "png"],
    help="Upload a clear image to run helmet and license plate detection."
)

if uploaded_file is not None:
    original_image = Image.open(uploaded_file).convert("RGB")

    st.image(
        original_image,
        caption="Uploaded Road Image",
        width=650
    )

    detect_clicked = st.button(
        "🔍 Detect Road Safety",
        use_container_width=True
    )

    if detect_clicked:
        api_key = get_api_key()

        if not api_key:
            st.error(
                "Roboflow API key not found. Add ROBOFLOW_API_KEY "
                "to Streamlit Secrets or your local environment."
            )
        else:
            try:
                with st.spinner("Analyzing image with RoadGuard AI..."):
                    client = InferenceHTTPClient(
                        api_url=API_URL,
                        api_key=api_key
                    )

                    import tempfile

                    # Use a temporary image file for inference.
                    with tempfile.NamedTemporaryFile(
                        suffix=".jpg", delete=False
                    ) as temp_file:
                        temp_path = temp_file.name
                        original_image.save(temp_path, format="JPEG")

                    try:
                        result = client.infer(
                            temp_path,
                            model_id=MODEL_ID
                        )
                    finally:
                        if os.path.exists(temp_path):
                            os.remove(temp_path)

                    predictions = result.get("predictions", [])
                    detected_image = draw_detections(
                        original_image, predictions
                    )
                    counts = count_predictions(predictions)

                    st.session_state["roadguard_result"] = {
                        "original": original_image.copy(),
                        "detected": detected_image,
                        "predictions": predictions,
                        "counts": counts
                    }

            except Exception as error:
                st.error(
                    "Detection failed. Check your Roboflow model, "
                    "API key, internet connection, and usage limits."
                )
                st.caption(f"Technical details: {error}")


# ---------------- DISPLAY RESULTS ----------------
saved_result = st.session_state.get("roadguard_result")

if saved_result is not None:
    st.markdown(
        '<div class="section-title">🖼️ Image Comparison</div>',
        unsafe_allow_html=True
    )

    left, right = st.columns(2, gap="large")

    with left:
        st.markdown("**Original Image**")
        st.image(
            saved_result["original"],
            use_container_width=True
        )

    with right:
        st.markdown("**Detection Image**")
        st.image(
            saved_result["detected"],
            use_container_width=True
        )

    # Full-width results below both images.
    st.markdown(
        '<div class="section-title">📊 Detection Result</div>',
        unsafe_allow_html=True
    )

    counts = saved_result["counts"]
    total = len(saved_result["predictions"])

    c1, c2, c3, c4 = st.columns(4, gap="medium")

    with c1:
        render_metric("🟢 Helmet", counts["helmet"], "#16A34A")

    with c2:
        render_metric("🔴 No Helmet", counts["no_helmet"], "#DC2626")

    with c3:
        render_metric("🔵 License Plate", counts["license_plate"], "#2563EB")

    with c4:
        render_metric("📍 Total Objects", total, "#315D96")

    if counts["no_helmet"] > 0:
        st.markdown(f"""
        <div class="status-box">
            <strong>⚠️ Safety Alert</strong><br>
            The model detected {counts["no_helmet"]} object(s)
            classified as no helmet. Please review the detection image.
        </div>
        """, unsafe_allow_html=True)
    elif total > 0:
        st.markdown("""
        <div class="status-box">
            <strong>✅ Detection Complete</strong><br>
            No objects classified as no helmet were detected in this image.
            This does not guarantee that the road scene is completely safe.
        </div>
        """, unsafe_allow_html=True)
    else:
        st.info(
            "No objects were detected. Try another clear image. "
            "The model may not recognize every object."
        )

    with st.expander("🔎 View Detection Details"):
        if saved_result["predictions"]:
            for index, prediction in enumerate(
                saved_result["predictions"], start=1
            ):
                label = prediction.get("class", "Unknown")
                confidence = float(
                    prediction.get("confidence", 0)
                )

                st.write(
                    f"**{index}. {label}** — "
                    f"Confidence: {confidence:.1%}"
                )
        else:
            st.write("No predictions were returned by the model.")


# ---------------- FOOTER ----------------
st.markdown("""
<div class="footer">
    RoadGuard AI · Intelligent Road Safety Monitoring<br>
    AI predictions are assistive and should be verified by a human.
</div>
""", unsafe_allow_html=True)
