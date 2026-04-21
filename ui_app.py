# uv pip install streamlit torch torchvision pillow requests matplotlib numpy

"""
Aplicație Streamlit pentru Clasificatorul de Fructe
===================================================
Include:
- Predicție imagine (Fișier local sau URL)
- Decizie inteligentă (Marjă de Încredere Automată)
- Explainable AI: Saliency Map
"""

import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import requests
from io import BytesIO
import matplotlib.pyplot as plt
import numpy as np

# ==========================================
# CONFIGURAȚIE
# ==========================================
MODEL_PATH = "best_ResNeXt.pth" 

CLASSES = [
    'apple fruit', 
    'banana fruit', 
    'cherry fruit', 
    'grapes fruit', 
    'orange fruit'
]

# ==========================================
# DEFINIREA ARHITECTURII (ResNeXt)
# ==========================================
class ResNeXtClassifier(nn.Module):
    def __init__(self, num_classes):
        super(ResNeXtClassifier, self).__init__()
        self.backbone = models.resnext50_32x4d(weights=None)
        num_features = self.backbone.fc.in_features
        
        self.backbone.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_features, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        return self.backbone(x)

# ==========================================
# FUNCȚII UTILITARE
# ==========================================
@st.cache_resource
def load_model(path):
    model = ResNeXtClassifier(num_classes=len(CLASSES))
    try:
        checkpoint = torch.load(path, map_location=torch.device('cpu'))
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        return model
    except Exception as e:
        st.error(f"Nu am putut încărca modelul de la {path}. Eroare: {e}")
        return None

def preprocess_image(image):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    return transform(image).unsqueeze(0)

def load_image_from_url(url):
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9"
        }
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status() 
        img = Image.open(BytesIO(response.content)).convert('RGB')
        return img
    except requests.exceptions.HTTPError as err_http:
        st.error(f"❌ Serverul nu permite descărcarea (Eroare HTTP: {err_http.response.status_code}). Încearcă să salvezi imaginea în PC și să o încarci manual.")
        return None
    except requests.exceptions.ConnectionError:
        st.error("❌ Nu m-am putut conecta la site. Verifică dacă link-ul este corect.")
        return None
    except Exception as e:
        st.error(f"❌ Imagine invalidă sau eroare necunoscută: {e}")
        return None

def generate_saliency_map(model, input_tensor, target_class_idx):
    """Calculează gradientul imaginii față de clasa prezisă."""
    model.eval()
    
    # Clonăm tensorul pentru a nu afecta alte operațiuni și activăm calculul de gradient
    img_tensor = input_tensor.clone()
    img_tensor.requires_grad = True
    
    # Forward pass
    output = model(img_tensor)
    score = output[0, target_class_idx]
    
    # Backward pass pentru a obține gradienții pixelilor
    model.zero_grad()
    score.backward()
    
    # Preluăm valoarea absolută maximă pe cele 3 canale (RGB)
    saliency = img_tensor.grad.data.abs().squeeze(0).max(0)[0]
    
    # Normalizăm între 0 și 1 pentru afișare corectă sub formă de heatmap
    saliency = (saliency - saliency.min()) / (saliency.max() - saliency.min() + 1e-8)
    
    return saliency.cpu().numpy()

# ==========================================
# INTERFAȚA STREAMLIT
# ==========================================
st.set_page_config(page_title="Clasificator Fructe", page_icon="🍎", layout="centered")
st.title("🍎🍌 Clasificator de Fructe Inteligent")
st.write("Sistemul se auto-reglează și include **Explainable AI (XAI)** pentru a-ți arăta cum gândește.")

model = load_model(MODEL_PATH)

if model is not None:
    input_method = st.radio("Sursa imaginii:", ("Încărcare de pe PC", "Link URL"))
    image = None
    
    if input_method == "Încărcare de pe PC":
        uploaded_file = st.file_uploader("Alege o imagine...", type=["jpg", "jpeg", "png", "webp"])
        if uploaded_file is not None:
            image = Image.open(uploaded_file).convert('RGB')
            
    elif input_method == "Link URL":
        url = st.text_input("Introdu URL-ul:")
        if url:
            image = load_image_from_url(url)

    if image is not None:
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("Imagine Analizată:")
            st.image(image, use_container_width=True)
            
        with col2:
            st.subheader("Rezultate:")
            with st.spinner('Procesare rețea neurală...'):
                input_tensor = preprocess_image(image)
                
                with torch.no_grad():
                    outputs = model(input_tensor)
                    probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
                
                probs_array = probabilities.numpy()
                sorted_indices = np.argsort(probs_array)[::-1]
                top1_idx = sorted_indices[0]
                top2_idx = sorted_indices[1]
                
                top1_name = CLASSES[top1_idx]
                top2_name = CLASSES[top2_idx]
                
                top1_conf = probs_array[top1_idx] * 100
                top2_conf = probs_array[top2_idx] * 100
                margin = top1_conf - top2_conf
                
                # Regula automată: Peste 50% general, și peste 15% distanță de concurent
                is_confident = (top1_conf >= 50.0) and (margin >= 15.0)
                
                if is_confident:
                    st.success(f"**Predicție:** {top1_name.upper()}")
                    st.metric(label="Încredere model", value=f"{top1_conf:.2f}%")
                else:
                    st.warning("🤔 **Nu sunt complet sigur.**")
                    st.write(f"*Pare a fi **{top1_name}** ({top1_conf:.1f}%), dar seamănă destul de mult și cu **{top2_name}** ({top2_conf:.1f}%).*")
                
        # ==========================================
        # VIZUALIZĂRI (Grafic Probabilități & XAI)
        # ==========================================
        st.markdown("---")
        
        # 1. Distribuția Probabilităților
        st.write("**Distribuția matematică a predicțiilor:**")
        fig_bar, ax_bar = plt.subplots(figsize=(8, 3))
        sorted_classes = [CLASSES[i] for i in sorted_indices]
        sorted_probs = [probs_array[i] * 100 for i in sorted_indices]
        colors = ['#2e7b32' if is_confident else '#ed6c02'] + ['skyblue'] * (len(CLASSES) - 1)
        
        bars = ax_bar.barh(sorted_classes, sorted_probs, color=colors)
        ax_bar.bar_label(bars, fmt='%.1f%%', padding=3)
        ax_bar.set_xlim(0, 110)
        ax_bar.set_xlabel('Încredere (%)')
        ax_bar.invert_yaxis() 
        ax_bar.spines['top'].set_visible(False)
        ax_bar.spines['right'].set_visible(False)
        st.pyplot(fig_bar)
        
        # 2. Saliency Map (Ce a văzut modelul?)
        st.write("**Explainable AI: Ce a contat cel mai mult pentru model?**")
        with st.spinner("Generez harta zonelor de interes (Saliency Map)..."):
            saliency = generate_saliency_map(model, input_tensor, top1_idx)
            
            # Redimensionăm imaginea originală la 224x224 (cum o vede modelul) pentru vizualizare
            img_resized = image.resize((224, 224))
            
            fig_xai, axes = plt.subplots(1, 3, figsize=(12, 4))
            
            # Imaginea Originală (văzută de model)
            axes[0].imshow(img_resized)
            axes[0].set_title('Cum vede modelul (224x224)', fontsize=10)
            axes[0].axis('off')
            
            # Heatmap (Saliency)
            axes[1].imshow(saliency, cmap='hot')
            axes[1].set_title(f'Focus pentru: {top1_name}', fontsize=10)
            axes[1].axis('off')
            
            # Suprapunere (Overlay)
            axes[2].imshow(img_resized)
            axes[2].imshow(saliency, cmap='hot', alpha=0.5)
            axes[2].set_title('Suprapunere (Original + Focus)', fontsize=10)
            axes[2].axis('off')
            
            plt.tight_layout()
            st.pyplot(fig_xai)
else:
    st.warning("Asigură-te că fișierul `best_ResNeXt.pth` este în același folder cu acest script.")