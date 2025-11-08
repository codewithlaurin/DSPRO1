import streamlit as st
from PIL import Image
import torch
from torchvision import models
import torch.nn as nn

# --- App title and description ---
st.set_page_config(page_title="Plant Disease Classifier", page_icon="🌱")
st.title("Plant Disease Classifier")
st.write("Upload a plant leaf image and get a quick prediction from your model.")

# --- Model Loader ---
@st.cache_resource
def load_model(model_path="plant_disease_model.pt"):
    """
    Loads a ResNet18 model with a custom classifier (27 classes)
    and applies the saved weights.
    """
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    model.fc = nn.Linear(model.fc.in_features, 27)

    state_dict = torch.load(model_path, map_location=torch.device("cpu"))
    model.load_state_dict(state_dict)

    model.eval()

    return model


model = load_model()

uploaded_file = st.file_uploader("Upload a plant leaf image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display image preview
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    # Placeholder for prediction button
    if st.button("🔍 Run Prediction"):
        st.info("Analyzing image...")

        from torchvision import transforms

        preprocess = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])
        ])

        input_tensor = preprocess(image).unsqueeze(0)

        with torch.no_grad():
            outputs = model(input_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            confidence, predicted_class = torch.max(probabilities, dim=1)

        # --- Class labels ---
        CLASS_NAMES = [
            "Apple___Apple_scab",
            "Apple___Black_rot",
            "Apple___Cedar_apple_rust",
            "Apple___healthy",
            "Tomato___Bacterial_spot",
            "Tomato___Early_blight",
            "Tomato___Late_blight",
            "Tomato___Leaf_Mold",
            "Tomato___Septoria_leaf_spot",
            "Tomato___Spider_mites Two-spotted_spider_mite",
            "Tomato___Target_Spot",
            "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
            "Tomato___Tomato_mosaic_virus",
            "Tomato___healthy",
            "Pepper,_bell___Bacterial_spot",
            "Pepper,_bell___healthy",
            "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)",
            "Grape___Black_rot",
            "Grape___Esca_(Black_Measles)",
            "Grape___healthy",
            "Corn_(maize)___Common_rust_",
            "Corn_(maize)___Northern_Leaf_Blight",
            "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot",
            "Corn_(maize)___healthy",
            "Potato___Late_blight",
            "Potato___Early_blight",
            "Potato___healthy"
        ]

        predicted_label = CLASS_NAMES[predicted_class.item()]
        confidence_score = confidence.item() * 100

        # --- Display result ---
        st.success(f"✅ Prediction: **{predicted_label}** ({confidence_score:.2f}% confidence)")
