import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, models
from PIL import Image
import numpy as np

# ---------------- 1. PAGE CONFIGURATION ----------------
st.set_page_config(page_title="PneumoScan AI", page_icon="🫁", layout="wide")

# Custom CSS for premium look
st.markdown("""
    <style>
    /* Gradient Background */
    .stApp {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
    }
    
    /* Title Styling */
    .title-text {
        color: #0e2a47;
        text-align: center;
        font-family: 'Helvetica Neue', sans-serif;
        font-weight: 700;
        margin-bottom: 0px;
    }
    .subtitle-text {
        color: #4a5568;
        text-align: center;
        font-size: 18px;
        margin-bottom: 30px;
    }
    
    /* Card Styling for columns */
    div[data-testid="column"] {
        background-color: white;
        padding: 20px;
        border-radius: 15px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    /* Info/Success box styling */
    .stAlert {
        border-radius: 10px;
    }
    </style>
    """, unsafe_allow_html=True)

# ---------------- 2. MODEL SETUP ----------------
def build_model(n_classes=2):
    m = models.resnet18(weights=None)
    m.fc = nn.Linear(m.fc.in_features, n_classes)
    return m

@st.cache_resource
def load_model():
    model = build_model(2)
    checkpoint = torch.load("pneumonia_resnet18_small.pth", map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint['state_dict'])
    model = model.float() 
    model.eval()
    return model

model = load_model()
device = torch.device('cpu')

eval_tf = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# ---------------- UPDATED OOD CHECK (Strict) ----------------
def is_xray(image):
    img_array = np.array(image.convert('L'))
    std_dev = img_array.std()
    mean_val = img_array.mean()
    
    # Strict conditions for X-ray
    if mean_val < 30 or mean_val > 210:
        return False
    if std_dev > 75: 
        return False
        
    return True

# ---------------- 3. UI DESIGN ----------------
st.markdown("<h1 class='title-text'>🫁 PneumoScan AI</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle-text'>Advanced Deep Learning System for Chest X-Ray Analysis</p>", unsafe_allow_html=True)

col1, col2 = st.columns([1, 1], gap="large")

with col1:
    st.markdown("### 📤 Upload X-Ray")
    uploaded_file = st.file_uploader("Choose a Chest X-Ray...", type=["jpg", "jpeg", "png"], label_visibility="collapsed")
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert('RGB')
        st.image(image, caption="Uploaded X-Ray Image", use_container_width=True)

with col2:
    st.markdown("### 📊 Diagnostic Report")
    
    if uploaded_file is not None:
        if not is_xray(image):
            st.error("❌ **INVALID IMAGE**\n\nThe uploaded image does not appear to be a Chest X-Ray. Please upload a valid medical X-Ray.")
        else:
            with st.spinner("AI is analyzing the scan..."):
                img_tensor = eval_tf(image).unsqueeze(0).to(device)
                with torch.no_grad():
                    output = F.softmax(model(img_tensor), dim=1)[0]
                
                prob_normal = output[0].item()
                prob_pneumonia = output[1].item()
                
                st.write("")
                if prob_pneumonia > 0.40:
                    st.error(f"⚠️ **PNEUMONIA DETECTED**")
                    st.metric(label="Confidence Score", value=f"{prob_pneumonia*100:.2f}%")
                    st.info("💡 **Recommendation:** Please consult a radiologist or a healthcare professional immediately for further evaluation.")
                else:
                    st.success(f"✅ **NORMAL**")
                    st.metric(label="Confidence Score", value=f"{prob_normal*100:.2f}%")
                    st.info("💡 **Recommendation:** The lungs appear clear. However, always consult a doctor for a final diagnosis.")
    else:
        st.info("👈 Please upload a Chest X-Ray on the left to see the AI diagnosis.")

st.write("---")
st.caption("Disclaimer: This AI tool is for educational purposes only and is not a substitute for professional medical advice.")
