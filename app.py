
import os

import streamlit as st
from PIL import Image, ImageDraw
from inference_sdk import InferenceHTTPClient


# ---------------- PAGE CONFIG ----------------

st.set_page_config(
    page_title="RoadGuard AI",
    page_icon="🛣️",
    layout="wide"
)


# ---------------- CUSTOM CSS ----------------

st.markdown(
    """
    <style>
    .main-title {
        text-align: center;
        font-size: 38px;
        font-weight: bold;
        color: #1677ff;
    }

    .subtitle {
        text-align: center;
        font-size: 17px;
        color: gray;
    }

    .summary-box {
        padding: 14px;
        border-radius: 10px;
        margin: 10px 0;
        font-size: 17px;
        font-weight: bold;
    }

    .safe-box {
        color: #008000;
        background-color: #e8f8e8;
        border-left: 6px solid #008000;
    }

    .danger-box {
        color: #dc2626;
        background-color: #fee2e2;
        border-left: 6px solid #dc2626;
    }

    .plate-box {
        color: #1677ff;
        background-color: #e8f1ff;
        border-left: 6px solid #1677ff;
    }
    </style>
    """,
    unsafe_allow_html=True
)


# ---------------- HEADER ----------------

st.markdown(
    '<div class="main-title">🛣️ RoadGuard AI</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Intelligent Road Safety Monitoring System'
    '</div>',
    unsafe_allow_html=True
)

st.divider()


# ---------------- API KEY ----------------

try:
    api_key = st.secrets.get("ROBOFLOW_API_KEY", "")
except Exception:
    api_key = ""

if not api_key:
    api_key = os.environ.get("ROBOFLOW_API_KEY", "")

if not api_key:
    st.error(
        "Roboflow API key not found. Configure "
        "ROBOFLOW_API_KEY in Streamlit Secrets "
        "or your environment variables."
    )
    st.stop()


# ---------------- ROBOFLOW CLIENT ----------------

@st.cache_resource
def get_client(key):
    return InferenceHTTPClient(
        api_url="https://serverless.roboflow.com",
        api_key=key
    )


client = get_client(api_key)

MODEL_ID = "helmet-and-no-helmet-rider-detection-8wgzg/2"


# ---------------- IMAGE UPLOAD ----------------

st.subheader("📤 Upload Road Image")

uploaded_file = st.file_uploader(
    "Choose a road image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    # Keep uploaded image on the LEFT at 300px width
    left_col, right_col = st.columns([1, 2])

    with left_col:
        st.subheader("Original Image")
        st.image(
            image,
            caption="Uploaded Road Image",
            width=300
        )

    # ---------------- DETECTION ----------------

    with st.spinner("Analyzing road image..."):

        try:
            predictions = client.infer(
                image,
                model_id=MODEL_ID
            )

            detections = predictions.get("predictions", [])

            result_image = image.copy()
            draw = ImageDraw.Draw(result_image)

            helmet_count = 0
            no_helmet_count = 0
            plate_count = 0

            # ---------------- DRAW BOXES ----------------

            for detection in detections:

                label = str(
                    detection.get("class", "unknown")
                )

                normalized_label = (
                    label.lower()
                    .strip()
                    .replace(" ", "_")
                    .replace("-", "_")
                )

                confidence = float(
                    detection.get("confidence", 0)
                )

                x = float(detection.get("x", 0))
                y = float(detection.get("y", 0))
                w = float(detection.get("width", 0))
                h = float(detection.get("height", 0))

                left = int(x - w / 2)
                top = int(y - h / 2)
                right = int(x + w / 2)
                bottom = int(y + h / 2)

                if normalized_label == "helmet":
                    box_color = "lime"
                    helmet_count += 1

                elif normalized_label == "no_helmet":
                    box_color = "red"
                    no_helmet_count += 1

                elif normalized_label == "license_plate":
                    box_color = "dodgerblue"
                    plate_count += 1

                else:
                    box_color = "yellow"

                # Bounding box
                draw.rectangle(
                    [left, top, right, bottom],
                    outline=box_color,
                    width=4
                )

                # Label and confidence
                draw.text(
                    (max(0, left), max(0, top - 22)),
                    f"{label}: {confidence:.0%}",
                    fill=box_color,
                    stroke_width=1,
                    stroke_fill="black"
                )

            # ---------------- RIGHT SIDE RESULTS ----------------

            with right_col:

                st.subheader("🖼️ Detection Image")

                st.image(
                    result_image,
                    caption="RoadGuard AI Detection Result",
                    width=300
                )

                st.subheader("📊 Detection Summary")

                total_detections = len(detections)

                c1, c2, c3 = st.columns(3)

                c1.metric("Helmet", helmet_count)
                c2.metric("No Helmet", no_helmet_count)
                c3.metric("License Plate", plate_count)

                st.metric(
                    "Total Detections",
                    total_detections
                )

                # Red warning for no-helmet detections
                if no_helmet_count > 0:

                    st.markdown(
                        f"""
                        <div class="summary-box danger-box">
                            🚨 NO HELMET DETECTED<br>
                            {no_helmet_count} no-helmet detection(s) found.
                            Always follow road safety rules.
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                elif helmet_count > 0:

                    st.markdown(
                        """
                        <div class="summary-box safe-box">
                            ✅ HELMET DETECTED<br>
                            Helmet detection found.
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                else:
                    st.info(
                        "Helmet status could not be determined "
                        "from this image."
                    )

                if plate_count > 0:
                    st.markdown(
                        f"""
                        <div class="summary-box plate-box">
                            🔵 LICENSE PLATE DETECTED<br>
                            {plate_count} license plate detection(s) found.
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                # ---------------- CONFIDENCE DETAILS ----------------

                if detections:
                    st.subheader("🔎 Detection Details")

                    for index, detection in enumerate(
                        detections, start=1
                    ):
                        label = str(
                            detection.get("class", "unknown")
                        )

                        confidence = float(
                            detection.get("confidence", 0)
                        )

                        st.write(
                            f"{index}. **{label}** — "
                            f"Confidence: **{confidence:.1%}**"
                        )
                else:
                    st.warning(
                        "No objects detected. Try another clear image."
                    )

        except Exception as error:
            st.error(
                "Detection failed. Check your internet connection, "
                "Roboflow API key, model ID, and deployment logs."
            )
            st.caption(f"Technical details: {error}")

else:
    st.info(
        "Upload a road image to start helmet and road safety detection."
    )

st.divider()

st.caption("RoadGuard AI | AI-assisted Road Safety Monitoring")
