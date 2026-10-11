# RoadGuard AI
## Intelligent Road Safety Monitoring

### Project Overview
RoadGuard AI is a Computer Vision project that detects helmet and no-helmet riders in road images using a trained object detection model.

### Objectives
- Detect helmet-wearing riders.
- Identify riders detected without helmets.
- Display bounding boxes around detected objects.
- Show confidence scores for predictions.

### Technologies Used
- Python
- Streamlit
- Roboflow Inference
- Pillow (PIL)

### Features
1. Upload JPG, JPEG, or PNG road images.
2. Detect helmets and no-helmet riders.
3. Display original and annotated images.
4. Show detection labels and confidence scores.
5. Display a detection summary.

### Model
Model ID: helmet-and-no-helmet-rider-detection-8wgzg/2

Expected classes:
- helmet
- no_helmet
- license_plate

### Installation
Install the required packages:

    python -m pip install streamlit pillow inference-sdk

### Run the Application

    python -m streamlit run app.py

### Workflow
1. Upload a road image.
2. Click Detect Helmet.
3. The trained model analyses the image.
4. View the annotated image and detection summary.

### Limitations
Detection depends on image quality and model performance. A missed detection does not guarantee that a rider is wearing a helmet. This prototype is intended for demonstration purposes.

### Future Enhancements
- Real-time video detection
- Detection statistics dashboard
- Improved detection under different lighting conditions

### Project Submission
HackZen 2.0 - Computer Vision Track

RoadGuard AI | Intelligent Road Safety Monitoring
