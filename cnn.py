import streamlit as st
import torch
from PIL import Image
from transformers import ViTImageProcessor, ViTForImageClassification  # Use ViTImageProcessor

# Load model and processor
@st.cache_resource
def load_model():
    model_name = "google/vit-base-patch16-224"
    model = ViTForImageClassification.from_pretrained(model_name)
    processor = ViTImageProcessor.from_pretrained(model_name)  # Use ViTImageProcessor
    return model, processor

# Prediction function
def classify(image, model, processor):
    image = image.convert("RGB")  # Ensure image is RGB
    inputs = processor(images=image, return_tensors="pt")
    pixel_values = inputs["pixel_values"].to(dtype=torch.float32)  # Convert to float32

    with torch.no_grad():
        outputs = model(pixel_values)  # Pass only pixel_values

    predicted_class = outputs.logits.argmax().item()  # Get predicted class index
    return model.config.id2label[predicted_class]  # Convert index to label

# Streamlit UI
st.title("Simple Hugging Face Image Classifier")
uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "png", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    model, processor = load_model()
    prediction = classify(image, model, processor)

    st.subheader(f"Prediction: {prediction}")
