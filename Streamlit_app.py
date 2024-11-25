import streamlit as st
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import cv2
import io
from io import BytesIO
# Modelinizi tanımlayın
class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        # Model mimarinizi burada tanımlayın

    def forward(self, x):
        # İleri doğru geçiş işlemini burada tanımlayın
        return x

# Modeli yükleyin
model_path = "best_model.pth"
model = MyModel()
model.load_state_dict(torch.load(model_path))
model.eval()

# Cihazı ayarlayın (GPU varsa kullan, yoksa CPU kullan)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Görüntüyü işleme fonksiyonu
def preprocess_image(img):
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    img = transform(img).unsqueeze(0)  # Batch boyutuna dönüştür
    return img

# Tahmin yapma fonksiyonu
def predict_image(img):
    img = preprocess_image(img)
    img = img.to(device)
    with torch.no_grad():
        outputs = model(img)
    _, predicted = torch.max(outputs, 1)
    return predicted.cpu().numpy()[0]

# Streamlit arayüzü
st.title("Fasulye Hastalığı Tespit Uygulaması")

camera_input = st.camera_input('Kameradan resim çek')
gallery_input = st.file_uploader('VEYA Fasulye Fotoğrafı Ekleyin', accept_multiple_files=False)

if camera_input is not None:
    img_bytes = camera_input.getvalue()
    img = Image.open(io.BytesIO(img_bytes))
    img_cv2 = np.array(img)

    predicted_class = predict_image(img_cv2)
    st.write(f"Tahmin Edilen Sınıf: {predicted_class}")

elif gallery_input is not None:
    img_bytes = gallery_input.getvalue()
    img = Image.open(io.BytesIO(img_bytes))
    img_cv2 = np.array(img)

    predicted_class = predict_image(img_cv2)
    st.write(f"Tahmin Edilen Sınıf: {predicted_class}")

else:
    st.write("Lütfen bir resim yükleyin veya kamera kullanarak bir resim çekin.")
