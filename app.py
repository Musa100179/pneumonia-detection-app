import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import numpy as np

class SimpleCNN(nn.Module):
    def __init__(self, num_classes=2):
        super(SimpleCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 28 * 28, 128), nn.ReLU(),
            nn.Linear(128, num_classes)
        )
    def forward(self, x):
        return self.classifier(self.features(x))

@st.cache_resource
def load_model():
    model = SimpleCNN(num_classes=2)
    model.load_state_dict(torch.load("pneumonia_model.pth", map_location=torch.device('cpu')))
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

def is_xray(image):
    img_array = np.array(image.convert('L'))
    std_dev = img_array.std()
    mean_val = img_array.mean()
    if mean_val > 180 or std_dev < 40: return False
    return True

st.title("🩺 AI Pneumonia Detection System")
st.write("Chest X-Ray upload karein. Model batayega ke Pneumonia hai ya nahi.")

uploaded_file = st.file_uploader("X-Ray image choose karein...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption="Uploaded Image", use_column_width=True)
    
    if not is_xray(image):
        st.error("❌ IRRELEVANT IMAGE: Yeh X-ray nahi lag rahi. Baraye meherbani chest X-ray upload karein.")
    else:
        img_tensor = eval_tf(image).unsqueeze(0).to(device)
        with torch.no_grad():
            output = F.softmax(model(img_tensor), dim=1)[0]
        
        prob_normal = output[0].item()
        prob_pneumonia = output[1].item()
        
        if prob_pneumonia > 0.5:
            st.warning(f"⚠️ PNEUMONIA DETECTED\nConfidence: {prob_pneumonia*100:.2f}%\nPlease consult a doctor.")
        else:
            st.success(f"✅ NORMAL\nConfidence: {prob_normal*100:.2f}%\nLungs look clear.")
