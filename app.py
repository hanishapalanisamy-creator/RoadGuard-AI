
import os
import streamlit as st
from PIL import Image, ImageDraw
from inference_sdk import InferenceHTTPClient

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="RoadGuard AI",
    page_icon="🛡️",
    layout="wide"
)

# ---------------- CUSTOM CSS ----------------
st.markdown("""
<style>
.stApp {
    background: #E9EEF5;
    color: #243B5A;
}
.block-container {
    padding-top: 1.2rem;
}
.roadguard-header {
    background: linear-gradient(110deg, #243B5A, #426B9B);
    padding: 22px 26px;
    border-radius: 14px;
    margin: 14px 0 22px;
}
.roadguard-header h1 {
    color: white;
    margin: 0;
    font-size: 30px;
}
.roadguard-header p {
    color: #E2EAF5;
    margin: 6px 0 0;
}
div.stButton > button,
div.stLinkButton > a {
    background: #315D96;
    color: white !important;
    border: 1px solid #315D96;
    border-radius: 8px;
    font-weight: 600;
    min-height: 40px;
}
div.stButton > button:hover,
div.stLinkButton > a:hover {
    background: #243B5A;
    color: white !important;
}
.result-card {
    background: white;
    border: 1px solid #D4DFEC;
    border-radius: 12px;
    padding: 14px 16px;
    margin: 8px 0 12px;
}
.result-card h4 {
    color: #243B5A;
    margin: 0;
}
.warning-box {
    background: #FEE2E2;
    color: #991B1B;
    border-left: 5px solid #DC2626;
    padding: 14px;
    border-radius: 8px;
    font-weight: 650;
    margin: 12px 0;
}
.safe-box {
    background: #DCFCE7;
    color: #166534;
    border-left: 5px solid #16A34A;
    padding: 14px;
    border-radius: 8px;
    font-weight: 650;
    margin: 12px 0;
}
.neutral-box {
    background: #DBEAFE;
    color: #1E40AF;
    border-left: 5px solid #2563EB;
    padding: 14px;
    border-radius: 8px;
    margin: 12px 0;
}
</style>
""", unsafe_allow_html=True)


# ---------------- SESSION STATE ----------------
if "show_help" not in st.session_state:
    st.session_state.show_help = False

if "show_share" not in st.session_state:
    st.session_state.show_share = False

if "result_data" not in st.session_state:
    st.session_state.result_data = None

if "result_signature" not in st.session_state:
    st.session_state.result_signature = None


# ---------------- NAVBAR ----------------
title_col, detect_col, help_col, share_col = st.columns(
    [5.2, 1.25, 1.0, 1.15],
    vertical_alignment="center"
)

with title_col:
    st.markdown("""
    <div style="padding:5px 0">
        <span style="font-size:27px">🛡️</span>
        <span style="font-size:25px;font-weight:750;color:#243B5A">
            RoadGuard AI
        </span>
    </div>
    """, unsafe_allow_html=True)

with detect_col:
    if st.button("🔍 Detect", use_container_width=True):
        st.session_state.show_help = False
        st.session_state.show_share = False

with help_col:
    if st.button("❔ Help", use_container_width=True):
        st.session_state.show_help = not st.session_state.show_help
        st.session_state.show_share = False

with share_col:
    if st.button("↗ Share", use_container_width=True):
        st.session_state.show_share = not st.session_state.show_share
        st.session_state.show_help = False


# ---------------- HELP MENU ----------------
if st.session_state.show_help:
    with st.container(border=True):
        st.markdown("### How to Use")
        st.markdown("""
        1. Upload a clear road image.
        2. Click **Detect Image**.
        3. Compare the original and detected images.
        4. Review the detection summary and safety warning.
        """)

        st.markdown("### Detection Instructions")
        st.markdown("""
        - 🟢 **Helmet:** A helmet was detected.
        - 🔴 **No Helmet:** A possible helmet violation was detected.
        - 🔵 **License Plate:** A license plate was detected.
        - ⚠️ Predictions may be incorrect. Verify results before taking action.
        """)


# ---------------- SHARE MENU ----------------
if st.session_state.show_share:
    with st.container(border=True):
        st.markdown("### Share RoadGuard AI")
        st.write("Share the project repository with others.")

        st.link_button(
            "Open GitHub Repository ↗",
            "https://github.com/hanishapalanisamy-creator/RoadGuard-AI",
            use_container_width=True
        )

        st.caption(
            "To share your live app, copy its URL from your browser."
        )


# ---------------- MAIN HEADER ----------------
st.markdown("""
<div class="roadguard-header">
    <h1>Intelligent Road Safety Monitoring</h1>
    <p>AI-powered helmet and license plate detection</p>
</div>
""", unsafe_allow_html=True)


# ---------------- ROBOFLOW CONFIG ----------------
MODEL_ID = "helmet-and-no-helmet-rider-detection-8wgzg/2"
API_URL = "https://serverless.roboflow.com"

api_key = st.secrets.get(
    "ROBOFLOW_API_KEY",
    os.getenv("ROBOFLOW_API_KEY", "")
)


# ---------------- HELPER FUNCTIONS ----------------
def normalize_label(label):
    return (
        str(label)
        .lower()
        .replace("_", " ")
        .replace("-", " ")
        .strip()
    )


def classify_label(label):
    name = normalize_label(label)

    if (
        "no helmet" in name
        or "without helmet" in name
        or "nohelmet" in name
    ):
        return "no_helmet"

    if (
        "license plate" in name
        or "number plate" in name
        or "licence plate" in name
    ):
        return "license_plate"

    if "helmet" in name:
        return "helmet"

    return "other"


def draw_detections(image, predictions):
    output = image.copy()
    draw = ImageDraw.Draw(output)

    colors = {
        "helmet": "#16A34A",
        "no_helmet": "#DC2626",
        "license_plate": "#2563EB",
        "other": "#F59E0B"
    }

    for pred in predictions:
        try:
            x = float(pred["x"])
            y = float(pred["y"])
            width = float(pred["width"])
            height = float(pred["height"])

            x1 = x - width / 2
            y1 = y - height / 2
            x2 = x + width / 2
            y2 = y + height / 2

            label = str(pred.get("class", "object"))
            category = classify_label(label)
            color = colors[category]

            confidence = float(pred.get("confidence", 0))
            caption = f"{label} {confidence:.0%}"

            draw.rectangle(
                (x1, y1, x2, y2),
                outline=color,
                width=4
            )

            bbox = draw.textbbox((0, 0), caption)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]

            label_top = max(0, y1 - text_height - 8)

            draw.rectangle(
                (
                    x1,
                    label_top,
                    x1 + text_width + 8,
                    label_top + text_height + 6
                ),
                fill=color
            )

            draw.text(
                (x1 + 4, label_top + 3),
                caption,
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

    for pred in predictions:
        category = classify_label(pred.get("class", ""))

        if category in counts:
            counts[category] += 1

    return counts


# ---------------- IMAGE UPLOAD ----------------
st.markdown(
    '<h3 style="color:#243B5A">Upload Road Image</h3>',
    unsafe_allow_html=True
)

uploaded_file = st.file_uploader(
    "Choose an image",
    type=["jpg", "jpeg", "png"],
    help="Upload a clear road image for detection."
)

if uploaded_file is not None:
    try:
        original_image = Image.open(uploaded_file).convert("RGB")
    except Exception:
        st.error("The selected file could not be opened as an image.")
        st.stop()

    # Clear stale results when the selected file changes.
    signature = (uploaded_file.name, uploaded_file.size)

    if st.session_state.result_signature != signature:
        st.session_state.result_data = None
        st.session_state.result_signature = signature

    if st.button(
        "🔍 Detect Image",
        key="run_detection",
        use_container_width=True
    ):
        if not api_key:
            st.error(
                "Roboflow API key is missing. Add ROBOFLOW_API_KEY "
                "in Streamlit Cloud Settings → Secrets."
            )
        else:
            try:
                with st.spinner("Analysing image with RoadGuard AI..."):
                    client = InferenceHTTPClient(
                        api_url=API_URL,
                        api_key=api_key
                    )

                    uploaded_file.seek(0)

                    response = client.infer(
                        uploaded_file,
                        model_id=MODEL_ID
                    )

                    predictions = response.get("predictions", [])

                    detected_image = draw_detections(
                        original_image,
                        predictions
                    )

                    counts = count_predictions(predictions)

                    st.session_state.result_data = {
                        "original": original_image.copy(),
                        "detected": detected_image,
                        "counts": counts
                    }

                    st.session_state.result_signature = signature

            except Exception as exc:
                st.error(
                    "Detection failed. Check your API key, model ID, "
                    "internet connection, and Roboflow model availability."
                )
                st.caption(f"Technical details: {exc}")


# ---------------- DISPLAY RESULTS ----------------
result = st.session_state.result_data

if result is not None:
    st.markdown("---")

    original_col, detected_col = st.columns(2, gap="medium")

    with original_col:
        st.markdown(
            '<div class="result-card"><h4>Original Image</h4></div>',
            unsafe_allow_html=True
        )
        st.image(
            result["original"],
            use_container_width=True
        )

    with detected_col:
        st.markdown(
            '<div class="result-card"><h4>Detected Image</h4></div>',
            unsafe_allow_html=True
        )
        st.image(
            result["detected"],
            use_container_width=True
        )

    counts = result["counts"]

    # ---------------- HELMET ALERT ----------------
    if counts["no_helmet"] > 0:
        st.markdown("""
        <div class="warning-box">
            ⚠️ Helmet Not Detected — Please wear a helmet for safety.
        </div>
        """, unsafe_allow_html=True)

    elif counts["helmet"] > 0:
        st.markdown("""
        <div class="safe-box">
            ✅ Helmet detected in the uploaded image.
            Please verify the model result.
        </div>
        """, unsafe_allow_html=True)

    else:
        st.markdown("""
        <div class="neutral-box">
            ℹ️ No helmet-related detection was found.
            This does not confirm that every rider is safe.
        </div>
        """, unsafe_allow_html=True)

    # ---------------- DETECTION SUMMARY ----------------
    st.markdown("""
    <div class="result-card">
        <h4>Detection Summary</h4>
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)

    col1.markdown(f"🟢 **Helmet:** {counts['helmet']}")
    col2.markdown(f"🔴 **No Helmet:** {counts['no_helmet']}")
    col3.markdown(f"🔵 **License Plate:** {counts['license_plate']}")
