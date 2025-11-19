# Rice-Leaf-Disease-Detection-Classification
🌾 Rice Leaf Disease Detection System
U-Net Segmentation + CNN Classification + Streamlit Web App

A complete deep-learning powered system for automatic detection, segmentation, and classification of rice leaf diseases.
This project uses a hybrid pipeline:

U-Net → detects leaf lesions

Rule-based logic → identifies Narrow Brown Spot

CNN classifier → predicts 4 major rice diseases

Streamlit UI → clean, user-friendly web interface

📸 Screenshots

Refer the Files in the repo

🧠 Features
✔ Accurate U-Net lesion segmentation

Identifies infected areas pixel-by-pixel.

✔ CNN-based disease classification

Supports the following diseases:

Bacterial Leaf Blight

Leaf Blast

Leaf Scald

Heath Blast

✔ Rule-based identification of Narrow Brown Spot

Uses lesion size thresholds to detect spot-type diseases not captured well by CNN.

✔ Healthy leaf detection

If U-Net finds no lesions → leaf is treated as healthy.

✔ Side-by-side result display

Uploaded image on the left → Predicted result on the right.

✔ Large, centered prediction label

Improves readability and visual clarity.

✔ Bounding boxes on infected areas

Highlights all lesion regions.

✔ End-to-end Streamlit web application

Intuitive interface for real-time inference.

📊 Model Performance

The CNN classifier achieved:

Test Accuracy: 83.27%

Per-class performance
Class	Precision	Recall	F1-score
Bacterial Leaf Blight	0.79	0.83	0.81
Leaf Blast	0.84	0.85	0.84
Leaf Scald	0.83	0.75	0.79
Heath Blast	0.88	0.90	0.89
🗂️ Project Structure
📦 rice-leaf-disease-detector
│
├── app.py                        # Main Streamlit application
├── unet80.h5                     # U-Net segmentation model
├── disease_classification.h5     # CNN classification model
├── requirements.txt              # Dependencies
├── README.md                     # Documentation
│
└── screenshots/                  # Place your UI screenshots here
    ├── upload_ui.png
    ├── predicted_output.png
    └── bounding_boxes.png

⚙️ Installation
1️⃣ Clone the repository
git clone https://github.com/<your-username>/rice-leaf-disease-detector.git
cd rice-leaf-disease-detector

2️⃣ Install required libraries
pip install -r requirements.txt

3️⃣ Add the trained models

Place the two model files in the project directory:

unet80.h5

disease_classification.h5

Update paths in app.py if needed.

▶️ Running the Web App
streamlit run app.py


The application will open automatically in your browser at:

http://localhost:8501


Upload an image to get:

Disease prediction

Segmented regions

Bounding boxes

Centered textual label

🔬 Detection Pipeline
1. Input Image

User uploads a rice leaf image.

2. U-Net Segmentation

Input: 256×256 image

Output: lesion mask

Contour extraction identifies infected regions.

3. Narrow Brown Spot Detection

If several small lesions exist (<1.2% area each), classify as NBS.

4. CNN Classification

Input: 224×224 leaf image

Output: one of 4 diseases

Softmax classifier

5. Result Visualization

Large centered label

Bounding boxes

Side-by-side UI display

🔧 Requirements

Add this to requirements.txt:

streamlit
tensorflow
numpy
opencv-python
pillow
scikit-learn

🚀 Future Enhancements

Switch to EfficientNetB0 for +10% accuracy

Add Grad-CAM visualization

Deploy to HuggingFace Spaces or Streamlit Cloud

Build dataset exploration dashboard

Improve color/spot-based detection for NBS
