import streamlit as st
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import cv2
from io import BytesIO

# Sınıf isimlerini tanımlayın
CLASS_NAMES = ["healthy", "angular_leaf_spot", "bean_rust"]

class_size = 3
model = models.efficientnet_v2_s(weights='DEFAULT')
model.classifier[1] = torch.nn.Linear(1280, class_size)

# Cihazı ayarlayın (GPU varsa kullan, yoksa CPU kullan)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Modeli yükleyin
model.load_state_dict(torch.load("best_model.pth", map_location=device))
model.eval()

# Görüntüyü işleme fonksiyonu
def preprocess_image(img):
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((256, 256)),
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
    probabilities = torch.nn.functional.softmax(outputs, dim=1)
    confidence, predicted = torch.max(probabilities, 1)
    print(predicted.cpu().numpy()[0])
    return predicted.cpu().numpy()[0], confidence.cpu().numpy()[0]

# Streamlit arayüzü
st.title("Fasulye Hastalığı Tespit Uygulaması")

camera_input = st.camera_input('Kameradan resim çek')
gallery_input = st.file_uploader('VEYA Fasulye Fotoğrafı Ekleyin', accept_multiple_files=False)

if camera_input is not None:
    img_bytes = camera_input.getvalue()
    img = Image.open(BytesIO(img_bytes))
    img_cv2 = np.array(img)

    predicted_class, confidence = predict_image(img_cv2)
    st.write(f"Tahmin Edilen Sınıf: {CLASS_NAMES[predicted_class]}")
    st.write(f"İnanılırlık Yüzdesi: {confidence*100:.2f}%")

elif gallery_input is not None:
    img_bytes = gallery_input.getvalue()
    img = Image.open(BytesIO(img_bytes))
    img_cv2 = np.array(img)

    predicted_class, confidence = predict_image(img_cv2)
    print(predicted_class)
    st.write(predicted_class)
    st.write(f"Tahmin Edilen Sınıf: {[predicted_class]}")
    st.write(f"İnanılırlık Yüzdesi: {confidence*100:.2f}%")

else:
    st.write("Lütfen bir resim yükleyin veya kamera kullanarak bir resim çekin.")
