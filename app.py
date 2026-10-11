
import os
import tempfile

import streamlit as st
from PIL import Image, ImageDraw
from inference_sdk import (
    InferenceHTTPClient,
    InferenceConfiguration
)

# Page configuration
st.set_page_config(
    page_title="RoadGuard AI",
    page_icon="🛡️",
    layout="wide"
)

# Project heading
st.title("🛡️ RoadGuard AI")
st.subheader("Intelligent Road Safety Monitoring")

st.write(
    "Upload a road image to test helmet and "
    "no-helmet rider detection."
)

# Image upload
uploaded_file = st.file_uploader(
    "Upload a road image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file).convert("RGB")

        col1, col2 = st.columns(2)

        # Original image
        with col1:
            st.subheader("📷 Uploaded Image")
            st.image(
                image,
                caption="Original Road Image",
                width=300
            )

        if st.button("🔍 Detect Helmet", type="primary"):

            # Read API key from Streamlit Secrets or environment
            try:
                api_key = st.secrets.get("ROBOFLOW_API_KEY", "")
            except Exception:
                api_key = ""

            if not api_key:
                api_key = os.getenv("ROBOFLOW_API_KEY", "")

            if not api_key:
                st.error(
                    "API key not found. Configure "
                    "ROBOFLOW_API_KEY in Streamlit Secrets "
                    "or your local environment."
                )
                st.stop()

            try:
                client = InferenceHTTPClient(
                    api_url="https://serverless.roboflow.com",
                    api_key=api_key
                ).configure(
                    InferenceConfiguration(
                        api_key_transport="header"
                    )
                )

                image_path = None

                try:
                    with tempfile.NamedTemporaryFile(
                        suffix=".jpg",
                        delete=False
                    ) as temp_file:
                        image.save(temp_file, format="JPEG")
                        image_path = temp_file.name

                    with st.spinner("Analysing image..."):
                        result = client.infer(
                            image_path,
                            model_id=(
                                "helmet-and-no-helmet-rider-detection-8wgzg/2"
                            )
                        )

                finally:
                    if image_path and os.path.exists(image_path):
                        os.remove(image_path)

                predictions = result.get("predictions", [])

                # Draw colour-coded detection boxes
                annotated = image.copy()
                draw = ImageDraw.Draw(annotated)

                for pred in predictions:
                    x = pred.get("x", 0)
                    y = pred.get("y", 0)
                    w = pred.get("width", 0)
                    h = pred.get("height", 0)

                    label = str(pred.get("class", "object"))
                    confidence = pred.get("confidence", 0)

                    left = x - w / 2
                    top = y - h / 2
                    right = x + w / 2
                    bottom = y + h / 2

                    # Match colour to the model's class label
                    normalized_label = label.lower().strip()

                    if normalized_label == "helmet":
                        box_color = "lime"
                    elif normalized_label == "no_helmet":
                        box_color = "red"
                    elif normalized_label == "license_plate":
                        box_color = "dodgerblue"
                    else:
                        box_color = "yellow"

                    draw.rectangle(
                        [left, top, right, bottom],
                        outline=box_color,
                        width=4
                    )

                    draw.text(
                        (left, max(0, top - 20)),
                        f"{label}: {confidence:.0%}",
                        fill=box_color,
                        stroke_width=1,
                        stroke_fill="black"
                    )

                # Detection image
                with col2:
                    st.subheader("✅ Detection Result")
                    st.image(
                        annotated,
                        caption="Helmet Detection Output",
                        width=300
                    )

                # Detection summary
                st.divider()
                st.subheader("📊 Detection Summary")

                if predictions:
                    st.success(
                        f"Model returned {len(predictions)} "
                        "detection(s)."
                    )

                    for index, pred in enumerate(
                        predictions, start=1
                    ):
                        label = pred.get("class", "Unknown")
                        confidence = pred.get("confidence", 0)

                        st.write(
                            f"{index}. **{label}** — "
                            f"{confidence:.1%} confidence"
                        )

                else:
                    st.warning(
                        "No objects detected in this image. "
                        "This does not guarantee that no "
                        "helmet or rider is present."
                    )

            except Exception:
                st.error(
                    "Detection failed. Check your API key, "
                    "model access, and connection. "
                    "See app logs for technical details."
                )

    except Exception:
        st.error(
            "Unable to open this image. Please upload "
            "a valid JPG, JPEG, or PNG image."
        )

# Colour legend
st.divider()
st.subheader("🎨 Detection Colour Guide")

st.markdown(
    """
    🟢 **Helmet** — Green box

    🔴 **No Helmet** — Red box

    🔵 **License Plate** — Blue box

    🟡 **Other detected object** — Yellow box
    """
)

# Footer
st.divider()
st.caption(
    "🛡️ RoadGuard AI | HackZen 2.0 Computer Vision Project"
)
