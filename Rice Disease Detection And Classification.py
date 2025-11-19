import streamlit as st
import numpy as np
import cv2
from PIL import Image
from tensorflow.keras.models import load_model

# =======================
# CONFIGURATION
# =======================
st.set_page_config(page_title="Rice Leaf Disease Detector", layout="wide")

st.title("🌾 Rice Leaf Disease Detection")
st.write("Detect rice leaf diseases using U-Net + CNN classifier.")

UNET_MODEL_PATH = "C:\\Users\\admin\\Downloads\\unet80.h5"
CLASSIFIER_MODEL_PATH = "C:\\Users\\admin\\Downloads\\disease_classification.h5"

@st.cache_resource
def load_models():
    unet_model = load_model(UNET_MODEL_PATH, compile=False)
    cls_model = load_model(CLASSIFIER_MODEL_PATH, compile=False)
    return unet_model, cls_model

unet_model, cls_model = load_models()

# Classifier labels
CLASS_NAMES = [
    "Bacterial Leaf Blight",
    "Leaf Blast",
    "Leaf Scald",
    "Heath Blast"
]

IMG_SIZE = 256


# =========================
# PROCESSING FUNCTION
# =========================
def detect_and_classify(
    img,
    unet_model,
    cls_model,
    threshold=0.4,
    small_spot_area_threshold=0.012,  # 1.2% area → narrow brown spot
    max_small_spots=3
):

    original_img = img.copy()

    # ----- Step 1: U-Net segmentation -----
    resized = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    inp = np.expand_dims(resized / 255.0, 0)
    pred_mask = unet_model.predict(inp, verbose=0)[0, :, :, 0]
    mask = (pred_mask > threshold).astype(np.uint8) * 255

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # If no contour → Healthy
    if len(contours) == 0:
        return "Healthy", original_img, []

    leaf_area = IMG_SIZE * IMG_SIZE
    small_spot_count = 0
    all_boxes = []

    # Collect bounding boxes
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        all_boxes.append((x, y, w, h))
        area_ratio = (w * h) / leaf_area

        if area_ratio < small_spot_area_threshold:
            small_spot_count += 1

    # ----- Step 2: Narrow Brown Spot -----
    if 1 <= small_spot_count <= max_small_spots:
        return "Narrow Brown Spot", original_img, all_boxes

    # ----- Step 3: Run classifier -----
    leaf_resized = cv2.resize(img, (224, 224)) / 255.0
    pred = cls_model.predict(np.expand_dims(leaf_resized, 0), verbose=0)
    label = CLASS_NAMES[np.argmax(pred)]

    return label, original_img, all_boxes


# =======================
# STREAMLIT UI
# =======================
uploaded_file = st.file_uploader("Upload a rice leaf image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    img_array = np.array(image)

    st.image(image, caption="Uploaded Leaf", use_column_width=True)

    st.write("🔍 Processing...")

    final_label, final_img, boxes = detect_and_classify(
        img=img_array,
        unet_model=unet_model,
        cls_model=cls_model
    )

    # Draw top label
    output = final_img.copy()
    cv2.putText(output, final_label, (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)

    # Draw boxes if not healthy
    if final_label != "Healthy":
        # Scale mask coords back to original image size
        h, w = final_img.shape[:2]
        scale_x = w / IMG_SIZE
        scale_y = h / IMG_SIZE

        for (x, y, bw, bh) in boxes:
            X1 = int(x * scale_x)
            Y1 = int(y * scale_y)
            X2 = int((x + bw) * scale_x)
            Y2 = int((y + bh) * scale_y)

            cv2.rectangle(output, (X1, Y1), (X2, Y2), (0, 255, 0), 3)

    st.image(output, caption=f"Final Result: {final_label}", use_column_width=True)
    st.success(f"✔ Disease Detected: {final_label}")
