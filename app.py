
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
    image = Image.open(uploaded_file).convert("RGB")

    # Two-column layout
    col1, col2 = st.columns(2)

    # Original image on the left
    with col1:
        st.subheader("📷 Uploaded Image")
        st.image(
            image,
            caption="Original Road Image",
             width=300
        )

    # Detection button
    if st.button("🔍 Detect Helmet", type="primary"):
        api_key = os.getenv("ROBOFLOW_API_KEY")

        if not api_key:
            st.error(
                "API key not found. Set "
                "ROBOFLOW_API_KEY in your terminal "
                "and restart the app."
            )

        else:
            try:
                client = InferenceHTTPClient(
                    api_url="https://serverless.roboflow.com",
                    api_key=api_key
                ).configure(
                    InferenceConfiguration(
                        api_key_transport="header"
                    )
                )

                # Save image temporarily for inference
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

                # Draw detection boxes
                annotated = image.copy()
                draw = ImageDraw.Draw(annotated)

                for pred in predictions:
                    x = pred.get("x", 0)
                    y = pred.get("y", 0)
                    w = pred.get("width", 0)
                    h = pred.get("height", 0)

                    label = pred.get("class", "object")
                    confidence = pred.get("confidence", 0)

                    left = x - w / 2
                    top = y - h / 2
                    right = x + w / 2
                    bottom = y + h / 2

                    draw.rectangle(
                        [left, top, right, bottom],
                        outline="lime",
                        width=4
                    )

                    draw.text(
                        (left, max(0, top - 20)),
                        f"{label}: {confidence:.0%}",
                        fill="lime",
                        stroke_width=1,
                        stroke_fill="black"
                    )

                # Detection image on the right
                with col2:
                    st.subheader("✅ Detection Result")
                    st.image(
                        annotated,
                        caption="Helmet Detection Output",
                         width=300
                    )

                # Results below both images
                st.divider()
                st.subheader("📊 Detection Summary")

                if predictions:
                    st.success(
                        f"Model returned {len(predictions)} "
                        "detection(s)."
                    )

                    for index, pred in enumerate(predictions, start=1):
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

            except Exception as error:
                st.error(
                    "Inference failed. Check your API key, "
                    "model access, and internet connection."
                )
                st.caption(str(error))

# Footer
st.divider()
st.caption(
    "🛡️ RoadGuard AI | HackZen 2.0 Computer Vision Project"
)
