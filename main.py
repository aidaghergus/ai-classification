"""
Clasificator de Fructe folosind PyTorch - VERSIUNE TEMĂ COMPLET DOCUMENTATĂ
===========================================================================
Acest script implementează un clasificator de imagini de fructe cu:
- 5 clase inițiale (extensibile la 9)
- Random Pixel Removal pentru occludere artificială
- Vizualizări before/after transform
- Analiza predicțiilor (top 5 corecte, greșite, borderline)
- Saliency Maps și SHAP pentru XAI (Explainable AI)
- Learning Rate Scheduler și Early Stopping
- Urmărire experimente prin MLflow cu comparare între rulări

Autor: AI Assistant
Data: 2026
"""

# ============================================================================
# IMPORTURI - Importăm toate bibliotecile necesare
# ============================================================================
import os             # Operatii pe sistemul de fișiere (căi, directoare)
import time           # Măsurarea timpului de antrenare
import random         # Generare numere aleatoare pentru seed și augmentări
import numpy as np    # Operatii numerice pe array-uri multidimensionale
from datetime import datetime  # Timestamp pentru salvare experimente
from collections import Counter  # Numărare frecvențe elemente

import torch          # Framework principal de Deep Learning
import torch.nn as nn  # Modulele neurale (straturi, funcții de activare, etc.)
import torch.optim as optim  # Optimizatori (SGD, Adam, etc.)
from torch.utils.data import Dataset, DataLoader, random_split  # Încărcare date
from torch.utils.tensorboard import SummaryWriter  # Logging pentru TensorBoard
import torchvision    # Module pre-antrenate și utilitare pentru vedere artificială
from torchvision import transforms, models  # Transformări imagini și modele pre-antrenate
from torchvision.datasets import ImageFolder  # Dataset din structura de directoare
from PIL import Image  # Procesare imagini
from sklearn.model_selection import StratifiedShuffleSplit
import mlflow         # Platformă de urmărire experimente ML
import mlflow.pytorch # Integrare MLflow cu PyTorch
from tqdm import tqdm  # Bare de progres pentru bucle
import matplotlib.pyplot as plt  # Vizualizări și grafice


def set_seed(seed=42):
    """
    Setează seed pentru toate bibliotecile pentru reproducibilitate.
    
    Această funcție asigură că rezultatele sunt reproductibile, adică
    de fiecare dată când rulăm codul, vom obține aceleași rezultate.
    Fără seed, inițializarea aleatoare a ponderilor ar duce la rezultate
    diferite la fiecare rulare.
    
    Parametri:
        seed (int): Valoarea seed-ului. Default 42 este o valoare comună
                   folosită în exemplele din documentația PyTorch.
    """
    random.seed(seed)                        # Seed pentru Python random
    np.random.seed(seed)                     # Seed pentru NumPy
    torch.manual_seed(seed)                  # Seed pentru PyTorch CPU
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)         # Seed pentru GPU actual
        torch.cuda.manual_seed_all(seed)     # Seed pentru toate GPU-urile
    # Următoarele două setări controlează comportamentul cuDNN pe GPU:
    # - deterministic=True: Folosește algoritmi deterministici (mai lent, dar reproductibil)
    # - benchmark=False: Nu caută cel mai rapid algoritm (pentru reproducibilitate)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ============================================================================
# CONFIGURATIE - Clasa care stochează toți hiperparametrii
# ============================================================================

class Config:
    """
    Clasa pentru configurarea hiperparametrilor și căilor.
    
    Această clasă centralizează toți parametrii configurabili ai sistemului,
    astfel încât să poată fi modificați ușor într-un singur loc.
    Hiperparametrii sunt setări care controlează procesul de antrenare și
    nu sunt învățate din date, ci sunt alese de noi.
    """
    
    # ---- Căi către directoare și fișiere ----
    DATA_DIR = "fruit_images"              # Directorul cu imaginile de fructe
    LOG_DIR = "runs/fruit_classifier"      # Director pentru log-uri TensorBoard
    MLFLOW_EXPERIMENT_NAME = "Fruit_Classification"  # Numele experimentului MLflow
    VIZ_DIR = "visualizations"             # Director pentru imaginile generate
    
    # ---- Lista claselor de fructe ----
    #SELECTED_CLASSES = [
    #    'apple fruit',      # 0 - Măr: fruct rotund, culoare roșie/verde
    #    'banana fruit',     # 1 - Banană: fruct alungit, culoare galbenă
    #    'cherry fruit',     # 2 - Cireșe: fruct mic, rotund, roșu
    #    'chickoo fruit',     # 3 - Chickoo:
    #    'grapes fruit',     # 4 - Struguri: ciorchine de boabe rotunde
    #    'kiwi fruit',     # 5 - Kiwi
    #    'mango fruit',     # 6 - Mango
    #    'orange fruit'      # 7 - Portocală: fruct rotund, portocaliu
    #    'strawberry fruit'      # 8 - Capsuna
    #]
    SELECTED_CLASSES = None  # Setează None pentru toate cele 9 clase
    
    # ---- Dimensiuni imagini ----
    IMG_SIZE = 224       # Dimensiunea la care redimensionăm imaginile (224x224 pixeli)
                         # Aceasta este dimensiunea standard pentru modelele pre-antrenate
                         # pe ImageNet (ResNet, VGG, etc.)
    
    BATCH_SIZE = 16      # Numărul de imagini procesate înainte de o actualizare a ponderilor
                         # 16 este un compromis bun între viteză și memorie
    
    # ---- Parametri antrenare ----
    EPOCHS = 10          # Numărul maxim de epoci
    LEARNING_RATE = 0.001  # Rata de învățare - Adam optimizer funcționează bine cu 1e-3
    TRAIN_SPLIT = 0.8    # 80% antrenare, 20% validare
    
    # ---- Parametri occludere ----
    OCCLUSION_PROB = 0.1  # 10% din pixeli eliminați aleatoriu
    
    # ---- Early Stopping - Previne overfitting ----
    EARLY_STOPPING_PATIENCE = 5     # Epoci fără îmbunătățire înainte de oprire
    EARLY_STOPPING_MIN_DELTA = 0.001  # Îmbunătățire minimă (0.1%)
    
    # ---- Device (CPU sau GPU) ----
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # ---- Parametri regularizare ----
    WEIGHT_DECAY = 1e-4  # Regularizare L2 pentru prevenirea overfitting
    DROPOUT_RATE = 0.5   # Dropout 50% pentru stratul fully-connected

    # ---- Grid Search Hiperparametri ----
    # Setează True pentru a rula grid search în loc de antrenarea standard
    HYPERPARAMETER_SEARCH = True
    SEARCH_MODEL = 'CNN'  # Modelul folosit pentru grid search (MLP / CNN / ResNeXt)
    SEARCH_BATCH_SIZES = [8, 32, 64]
    SEARCH_LEARNING_RATES = [0.0001, 0.01]
    SEARCH_EPOCHS = [10, 30]

    # ---- Comparare transformări suplimentare ----
    # Setează True pentru a rula comparația cu/fără RandomGrayscale + GaussianBlur
    COMPARE_TRANSFORMS = True
    # Când False, get_transforms() folosește configurația curentă (USE_EXTRA_TRANSFORMS)
    USE_EXTRA_TRANSFORMS = True  # Activează RandomGrayscale + GaussianBlur
    
    @classmethod
    def print_config(cls):
        """Afișează configurația curentă pentru informare."""
        print("=" * 60)
        print("CONFIGURATIE")
        print("=" * 60)
        print(f"Device: {cls.DEVICE}")
        print(f"Dimensiune imagine: {cls.IMG_SIZE}x{cls.IMG_SIZE}")
        print(f"Batch size: {cls.BATCH_SIZE}")
        print(f"Epoci: {cls.EPOCHS}")
        print(f"Learning rate: {cls.LEARNING_RATE}")
        print(f"Early Stopping Patience: {cls.EARLY_STOPPING_PATIENCE}")
        print(f"Split antrenare/validare: {cls.TRAIN_SPLIT}/{1-cls.TRAIN_SPLIT}")
        if cls.SELECTED_CLASSES:
            print(f"Clase selectate: {len(cls.SELECTED_CLASSES)}")
            print(f"  {cls.SELECTED_CLASSES}")
        else:
            print(f"Clase: TOATE (9)")
        print("=" * 60)


# ============================================================================
# TRANSFORMĂRI CUSTOM
# ============================================================================

class RandomPixelRemoval:
    """
    Transformare custom care simulează ocluzia prin eliminarea aleatorie de pixeli.
    
    Această tehnică este similară cu Dropout, dar aplicată direct pe imaginile
    de intrare. Efecte:
    - Previne overfitting-ul prin forțarea modelului de a nu depinde de anumiți pixeli
    - Simulează condiții reale de ocluzie (fruct parțial ascuns)
    - Determină modelul să învețe caracteristici distribuite, nu localizate
    """
    
    def __init__(self, probability=0.1):
        """
        Inițializează transformarea.
        
        Parametri:
            probability (float): Probabilitatea ca un pixel să fie eliminat.
                                - 0.1 = 10% pixeli eliminați
                                - 0.0 = niciun pixel eliminat
                                - 0.5 = jumătate pixeli eliminați (agresiv)
        """
        self.probability = probability
    
    def __call__(self, tensor):
        """
        Aplică eliminarea aleatorie de pixeli pe un tensor PyTorch.
        
        Parametri:
            tensor (torch.Tensor): Tensor de intrare cu shape (C, H, W)
        
        Returnări:
            torch.Tensor: Tensor cu același shape, dar cu unele valori zero
        """
        # Creează mască aleatorie: True pentru pixelii păstrați
        mask = torch.rand_like(tensor) > self.probability
        # Aplică masca: pixelii cu False devin 0
        return tensor * mask
    
    def __repr__(self):
        """Returnează reprezentarea ca string pentru debugging."""
        return f"{self.__class__.__name__}(probability={self.probability})"


def get_transforms(phase='train'):
    """
    Creează pipeline de transformări pentru imagini.
    
    Pipeline-urile de transformări sunt lanțuri de operații aplicate consecutiv.
    Faza de antrenare include augmentări, faza de validare doar preprocesare.
    
    Parametri:
        phase (str): 'train' pentru antrenare sau 'val' pentru validare
                    
    Returnări:
        transforms.Compose: Pipeline de transformări
    """
    
    if phase == 'train':
        # ---- TRANSFORMĂRI DE ANTRENARE (Data Augmentation) ----
        # Aceste transformări creează variații artificiale ale imaginilor pentru
        # a crește dimensiunea efectivă a dataset-ului și a îmbunătăți generalizarea.
        
        transform_list = [
            # PASUL 1: Redimensionare la 224x224 (standard pentru ImageNet)
            transforms.Resize((Config.IMG_SIZE, Config.IMG_SIZE)),
            
            # PASUL 2: Augmentări geometrice - simulează poziții diferite
            transforms.RandomHorizontalFlip(p=0.5),
            # Oglindire orizontală cu probabilitatea 50%
            # Utile pentru obiecte simetrice (mere, portocale)
            
            # transforms.RandomVerticalFlip(p=0.3),
            # Oglindire verticală cu probabilitatea 30%
            # Probabilitate mai mică deoarece fructele nu sunt de obicei "răsturnate"
            
            transforms.RandomRotation(degrees=30),
            # Rotire aleatorie [-30, +30] grade
            # Ajută modelul să recunoască fructul din orice orientare
            
            # PASUL 3: Transformare Afină - zoom, translație, forfecare
            # transforms.RandomAffine(
            #     degrees=0,              # Fără rotație suplimentară
            #     translate=(0.1, 0.1),   # Translație max 10% pe fiecare axă
            #     scale=(0.9, 1.1),       # Zoom între 90%-110%
            #     shear=(-5, 5)           # Forfecare [-5, +5] grade
            # ),
            
            # PASUL 4: Perspectivă - simulare unghiuri de cameră
            transforms.RandomPerspective(distortion_scale=0.2, p=0.3),
            # distortion_scale=0.2: distorsiune ușoară
            # p=0.3: aplicată pe 30% din imagini
            
            # PASUL 5: Crop aleator cu resize - simulare zoom digital
            # transforms.RandomResizedCrop(
            #     size=224,
            #     scale=(0.8, 1.0),       # Crop între 80%-100% din imagine
            #     ratio=(0.9, 1.1)        # Raport aspect variabil
            # ),
            
            # PASUL 6: Augmentări de culoare - simulează iluminare diferită
            transforms.ColorJitter(
                brightness=0.2,         # Variație luminozitate ±20%
                contrast=0.2,           # Variație contrast ±20%
                saturation=0.2,         # Variație saturație ±20%
                hue=0.1                 # Variație nuanță ±10%
            ),
            
            # PASUL 7: Grayscale aleatoriu - elimină dependența de culoare
            # 10% din imagini devin gri - forțează modelul să învețe forma/textura,
            # nu doar culoarea (util: kiwi verde vs. măr verde - culoarea nu ajută)
            # PASUL 8: Gaussian Blur - simulare neclaritate
            # kernel_size=5: matrice 5x5 de mediere ponderată
            # sigma=(0.1, 2.0): intensitatea blur-ului variază aleatoriu
            # Ambele transformări sunt controlate de Config.USE_EXTRA_TRANSFORMS
        ]

        if Config.USE_EXTRA_TRANSFORMS:
            transform_list += [
                transforms.RandomGrayscale(p=0.1),
                transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0)),
            ]

        transform_list += [
            
            # PASUL 9: Conversie la Tensor și Normalizare ImageNet
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],  # Mediile ImageNet pentru R, G, B
                std=[0.229, 0.224, 0.225]    # Deviațiile standard ImageNet
                # Formula: output = (input - mean) / std
                # Necesare pentru modele pre-antrenate pe ImageNet
            ),
            
            # PASUL 10: Occludere artificială - Random Pixel Removal
            # RandomPixelRemoval(probability=Config.OCCLUSION_PROB)
            # Elimină 10% din pixeli pentru a simula ocluzia
        ]
        
        transform = transforms.Compose(transform_list)
        
    else:
        # ---- TRANSFORMĂRI DE VALIDARE (fără augmentare) ----
        # Pentru validare, NU aplicăm augmentări deoarece vrem o evaluare
        # consistentă și reproductibilă a modelului.
        
        transform = transforms.Compose([
            transforms.Resize((Config.IMG_SIZE, Config.IMG_SIZE)),
            # Redimensionare la aceeași dimensiune ca la antrenare
            
            transforms.ToTensor(),
            # Conversie la tensor
            
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
                # ACEEAȘI normalizare ca la antrenare - CRITIC!
            )
            # NU aplicăm RandomPixelRemoval la validare
        ])
    
    return transform


# ============================================================================
# DATASET CUSTOM - Filtrare clase
# ============================================================================

class FilteredDataset(torch.utils.data.Dataset):
    """
    Dataset wrapper care filtrează clasele și remap-ează etichetele.
    
    Necesara deoarece:
    1. ImageFolder încarcă TOATE clasele din director
    2. Noi vrem doar anumite clase (Config.SELECTED_CLASSES)
    3. Etichetele trebuie remapate: 0, 1, 2, 3, 4 pentru clasele selectate
    """
    
    def __init__(self, base_dataset, selected_classes, original_classes):
        """
        Inițializează dataset-ul filtrat.
        
        Parametri:
            base_dataset (ImageFolder): Dataset-ul original
            selected_classes (list): Clasele dorite
            original_classes (list): Toate clasele din ImageFolder
        """
        self.base_dataset = base_dataset
        self.selected_classes = selected_classes
        self.original_classes = original_classes
        
        # Mapare nume_clasă -> etichetă_nouă (0, 1, 2, ...)
        self.class_to_idx = {
            cls_name: idx 
            for idx, cls_name in enumerate(selected_classes)
        }
        
        # Filtrează și remap-ează etichetele
        self.samples = []
        for idx, (path, original_label) in enumerate(base_dataset.samples):
            class_name = original_classes[original_label]
            if class_name in selected_classes:
                new_label = self.class_to_idx[class_name]
                self.samples.append((path, new_label))
        
        self.transform = base_dataset.transform
    
    def __len__(self):
        """Returnează numărul de imagini din dataset-ul filtrat."""
        return len(self.samples)
    
    def __getitem__(self, idx):
        """
        Returnează o imagine și eticheta la indexul specificat.
        
        Parametri:
            idx (int): Indexul imaginii
            
        Returnări:
            tuple: (imagine_transformata, eticheta)
        """
        path, target = self.samples[idx]
        sample = Image.open(path).convert('RGB')
        # convert('RGB') asigură 3 canale (unele imagini pot fi RGBA sau grayscale)
        if self.transform is not None:
            sample = self.transform(sample)
        return sample, target


def load_data():
    """
    Încarcă datele și creează DataLoaders.
    
    Returnări:
        tuple: (train_loader, val_loader, num_classes, class_names)
    """
    print("\n[DATA] Încărcare date...")
    
    # ImageFolder încarcă imagini din structura: root/class_name/image.jpg
    full_dataset = ImageFolder(
        root=Config.DATA_DIR,
        transform=get_transforms('train')
    )
    
    # Filtrare clase dacă am specificat
    if Config.SELECTED_CLASSES:
        full_dataset = FilteredDataset(
            full_dataset, 
            Config.SELECTED_CLASSES, 
            full_dataset.classes
        )
        class_names = Config.SELECTED_CLASSES
        num_classes = len(class_names)
        print(f"[DATA] Folosind {num_classes} clase selectate")
    else:
        class_names = full_dataset.classes
        num_classes = len(class_names)
        print(f"[DATA] Folosind toate cele {num_classes} clase")
    
    print(f"[DATA] Total imagini: {len(full_dataset)}")
    print(f"[DATA] Clase: {class_names}")
    
    # Split train/validation
    train_size = int(Config.TRAIN_SPLIT * len(full_dataset))
    val_size = len(full_dataset) - train_size
    
    # random_split cu seed fix pentru reproducibilitate
    # train_dataset, val_dataset = random_split(
    #     full_dataset, 
    #     [train_size, val_size],
    #     generator=torch.Generator().manual_seed(42)
    # )
    # Folosim StratifiedShuffleSplit pentru a păstra distribuția claselor în train/val
    sss = StratifiedShuffleSplit(n_splits=1, test_size=val_size, random_state=42)
    indices = list(range(len(full_dataset)))
    targets = [full_dataset.samples[i][1] for i in indices]  # Etichetele originale
    train_indices, val_indices = next(sss.split(indices, targets))
    train_dataset = torch.utils.data.Subset(full_dataset, train_indices)
    val_dataset = torch.utils.data.Subset(full_dataset, val_indices)


    # Pentru validare, folosim transformări fără augmentare
    val_dataset_full = ImageFolder(
        root=Config.DATA_DIR,
        transform=get_transforms('val')
    )
    
    if Config.SELECTED_CLASSES:
        val_dataset_full = FilteredDataset(
            val_dataset_full,
            Config.SELECTED_CLASSES,
            val_dataset_full.classes
        )
    
    # Păstrăm aceiași indici de validare
    val_indices = val_dataset.indices if hasattr(val_dataset, 'indices') else list(range(len(full_dataset) - val_size, len(full_dataset)))
    val_dataset = torch.utils.data.Subset(val_dataset_full, val_indices)
    
    print(f"[DATA] Train: {len(train_dataset)}, Val: {len(val_dataset)}")
    
    # DataLoader pentru antrenare - shuffle=True pentru amestecare
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,           # Amestecă la fiecare epocă - IMPORTANT!
        num_workers=0,
        pin_memory=torch.cuda.is_available()  # Accelerare transfer CPU->GPU
    )
    
    # DataLoader pentru validare - shuffle=False pentru consistență
    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,          # Nu amestecăm la validare
        num_workers=0,
        pin_memory=torch.cuda.is_available()
    )
    
    return train_loader, val_loader, num_classes, class_names


# ============================================================================
# VIZUALIZĂRI
# ============================================================================

def visualize_transforms(num_images=6, save_path=None):
    """
    Vizualizează imaginile înainte și după transformări.
    
    Parametri:
        num_images (int): Câte imagini să vizualizăm
        save_path (str): Calea unde să salvăm (opțional)
    """
    print("\n[VIZ] Generare vizualizare transformări...")
    os.makedirs(Config.VIZ_DIR, exist_ok=True)
    
    class_folders = Config.SELECTED_CLASSES if Config.SELECTED_CLASSES else os.listdir(Config.DATA_DIR)
    sample_images = []
    
    for class_name in class_folders[:3]:
        class_path = os.path.join(Config.DATA_DIR, class_name)
        if os.path.isdir(class_path):
            images = os.listdir(class_path)[:2]
            for img_name in images:
                sample_images.append(os.path.join(class_path, img_name))
        if len(sample_images) >= num_images:
            break
    
    sample_images = sample_images[:num_images]
    
    transform_train = get_transforms('train')
    transform_val = get_transforms('val')
    
    fig, axes = plt.subplots(num_images, 3, figsize=(12, 4*num_images))
    if num_images == 1:
        axes = axes.reshape(1, -1)
    
    # Parametri pentru denormalizare
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    
    for idx, img_path in enumerate(sample_images):
        img_original = Image.open(img_path).convert('RGB')
        img_train = transform_train(img_original)
        img_val = transform_val(img_original)
        
        # Denormalizare pentru vizualizare
        img_train_vis = torch.clamp(img_train * std + mean, 0, 1)
        img_val_vis = torch.clamp(img_val * std + mean, 0, 1)
        
        axes[idx, 0].imshow(img_original)
        axes[idx, 0].set_title('Original')
        axes[idx, 0].axis('off')
        
        axes[idx, 1].imshow(img_train_vis.permute(1, 2, 0).numpy())
        axes[idx, 1].set_title('Train Transform')
        axes[idx, 1].axis('off')
        
        axes[idx, 2].imshow(img_val_vis.permute(1, 2, 0).numpy())
        axes[idx, 2].set_title('Val Transform')
        axes[idx, 2].axis('off')
    
    plt.tight_layout()
    if save_path is None:
        save_path = os.path.join(Config.VIZ_DIR, 'transforms_comparison.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"[VIZ] Salvat în: {save_path}")
    # plt.show()
    plt.close()


def analyze_predictions(model, val_loader, class_names, device, num_samples=5):
    """
    Analizează predicțiile modelului pe datele de validare.
    
    Identifică:
    1. Predicții corecte cu încredere mare
    2. Predicții greșite cu încredere mare
    3. Predicții borderline (incerte)
    """
    print("\n[VIZ] Analiză predicții...")
    model.eval()
    predictions_data = []
    
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            confidences, predicted = torch.max(probabilities, 1)
            
            # Diferența dintre primele 2 predicții (mică = incert)
            top2_probs, top2_indices = torch.topk(probabilities, 2, dim=1)
            prob_diff = top2_probs[:, 0] - top2_probs[:, 1]
            
            for i in range(len(images)):
                predictions_data.append({
                    'image': images[i].cpu(),
                    'true_label': labels[i].item(),
                    'pred_label': predicted[i].item(),
                    'confidence': confidences[i].item(),
                    'prob_diff': prob_diff[i].item(),
                    'correct': predicted[i] == labels[i],
                    'top2_probs': top2_probs[i].cpu().numpy(),
                    'top2_indices': top2_indices[i].cpu().numpy()
                })
    
    # Categorii de predicții
    correct_preds = [p for p in predictions_data if p['correct']]
    wrong_preds = [p for p in predictions_data if not p['correct']]
    
    top_correct = sorted(correct_preds, key=lambda x: x['confidence'], reverse=True)[:num_samples]
    top_wrong = sorted(wrong_preds, key=lambda x: x['confidence'], reverse=True)[:num_samples]
    top_borderline = sorted(predictions_data, key=lambda x: x['prob_diff'])[:num_samples]
    
    # Vizualizare
    fig, axes = plt.subplots(3, num_samples, figsize=(3*num_samples, 9))
    categories = [
        ('Corecte (Top Încredere)', top_correct),
        ('Greșite (Top Încredere)', top_wrong),
        ('Borderline (Incerte)', top_borderline)
    ]
    
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    
    for row, (title, preds) in enumerate(categories):
        for col, pred in enumerate(preds):
            img = torch.clamp(pred['image'] * std + mean, 0, 1).permute(1, 2, 0).numpy()
            axes[row, col].imshow(img)
            
            true_name = class_names[pred['true_label']]
            pred_name = class_names[pred['pred_label']]
            conf = pred['confidence'] * 100
            
            if row == 2:
                # Borderline: afișăm ambele predicții
                second_class = class_names[pred['top2_indices'][1]]
                second_prob = pred['top2_probs'][1] * 100
                axes[row, col].set_title(
                    f'Adevărat: {true_name}\nPred: {pred_name} ({conf:.1f}%)\n#2: {second_class} ({second_prob:.1f}%)',
                    fontsize=9
                )
            else:
                color = 'green' if pred['correct'] else 'red'
                axes[row, col].set_title(
                    f'Adevărat: {true_name}\nPred: {pred_name}\nÎncredere: {conf:.1f}%',
                    fontsize=9, color=color
                )
            axes[row, col].axis('off')
        
        axes[row, 0].set_ylabel(title, fontsize=11, fontweight='bold', rotation=90, labelpad=20)
    
    plt.suptitle('Analiza Predicțiilor', fontsize=14, fontweight='bold')
    plt.tight_layout()
    save_path = os.path.join(Config.VIZ_DIR, 'predictions_analysis.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"[VIZ] Salvat în: {save_path}")
    #plt.show()
    plt.close()
    return top_correct, top_wrong, top_borderline


def generate_saliency_map(model, image, target_class, device):
    """
    Generează Saliency Map folosind metoda bazată pe gradient.
    
    Saliency Maps arată care pixeli din imagine au contribuit cel mai mult
    la decizia modelului. Se calculează gradientul ieșirii față de intrare.
    
    Parametri:
        model: Modelul antrenat
        image: Imaginea de analizat
        target_class: Clasa pentru care calculăm saliency
        device: CPU sau GPU
        
    Returnări:
        tuple: (imaginea_originala, harta_saliency)
    """
    model.eval()
    image = image.clone().unsqueeze(0).to(device)
    image.requires_grad = True  # Necesită gradient pentru calculul backpropagation
    
    # Forward pass
    output = model(image)
    score = output[0, target_class]  # Scorul pentru clasa țintă
    
    # Backward pass - calculăm gradientul scorului față de pixelii de intrare
    model.zero_grad()
    score.backward()
    
    # Saliency = valoarea absolută maximă a gradientului pe canale
    saliency = image.grad.data.abs().squeeze(0).max(0)[0]
    # Normalizare la [0, 1]
    saliency = (saliency - saliency.min()) / (saliency.max() - saliency.min() + 1e-8)
    
    return image.detach().squeeze(0), saliency


def visualize_saliency_maps(model, val_loader, class_names, device, num_samples=5):
    """
    Vizualizează Saliency Maps pentru a înțelege deciziile modelului.
    """
    print("\n[XAI] Generare Saliency Maps...")
    model.eval()
    samples = []
    
    with torch.no_grad():
        for images, labels in val_loader:
            for i in range(min(len(images), num_samples - len(samples))):
                samples.append((images[i], labels[i]))
            if len(samples) >= num_samples:
                break
    
    fig, axes = plt.subplots(num_samples, 3, figsize=(12, 4*num_samples))
    if num_samples == 1:
        axes = axes.reshape(1, -1)
    
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    
    for idx, (image, label) in enumerate(samples):
        # Predict without grad
        with torch.no_grad():
            output = model(image.unsqueeze(0).to(device))
            pred_label = output.argmax(1).item()
        
        # Generate saliency map
        image_grad, saliency = generate_saliency_map(model, image, pred_label, device)
        img_vis = torch.clamp(image * std + mean, 0, 1).permute(1, 2, 0).numpy()
        
        axes[idx, 0].imshow(img_vis)
        axes[idx, 0].set_title(f'Original\n{class_names[label]}')
        axes[idx, 0].axis('off')
        
        # Saliency heatmap
        axes[idx, 1].imshow(saliency.cpu().numpy(), cmap='hot')
        axes[idx, 1].set_title(f'Saliency\nPred: {class_names[pred_label]}')
        axes[idx, 1].axis('off')
        
        # Overlay
        axes[idx, 2].imshow(img_vis)
        axes[idx, 2].imshow(saliency.cpu().numpy(), cmap='hot', alpha=0.5)
        axes[idx, 2].set_title('Overlay')
        axes[idx, 2].axis('off')
    
    plt.suptitle('Saliency Maps - Ce vede modelul', fontsize=14, fontweight='bold')
    plt.tight_layout()
    save_path = os.path.join(Config.VIZ_DIR, 'saliency_maps.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"[XAI] Salvat în: {save_path}")
    #plt.show()
    plt.close()


def explain_with_shap(model, val_loader, class_names, device, num_samples=5):
    """
    Generează explicații SHAP (SHapley Additive exPlanations).
    
    SHAP folosește teoria jocurilor pentru a atribui fiecărui feature (pixel)
    o importanță pentru predicția finală. Este o metodă mai avansată de XAI.
    """
    try:
        import shap
    except ImportError:
        print("\n[XAI] SHAP nu este instalat. Instalează cu: pip install shap")
        return
    
    print("\n[XAI] Generare SHAP...")
    model.eval()
    
    # Pregătim background și imagini de test
    background_images = []
    test_images = []
    
    with torch.no_grad():
        for images, labels in val_loader:
            if len(background_images) < 10:
                background_images.append(images[:min(10-len(background_images), len(images))])
            if len(test_images) < num_samples:
                remaining = num_samples - len(test_images)
                test_images.extend([(img, lbl) for img, lbl in zip(images[:remaining], labels[:remaining])])
            if len(background_images) >= 10 and len(test_images) >= num_samples:
                break
    
    background = torch.cat(background_images, dim=0).to(device)
    test_imgs = torch.stack([img for img, _ in test_images]).to(device)
    test_labels = [lbl.item() for _, lbl in test_images]
    
    # SHAP DeepExplainer
    explainer = shap.DeepExplainer(model, background)
    shap_values = explainer.shap_values(test_imgs)
    
    # Denormalizare pentru vizualizare
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1).to(device)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1).to(device)
    test_imgs_denorm = torch.clamp(test_imgs * std + mean, 0, 1)
    
    fig, axes = plt.subplots(num_samples, 2, figsize=(10, 5*num_samples))
    if num_samples == 1:
        axes = axes.reshape(1, -1)
    
    for idx in range(num_samples):
        img = test_imgs_denorm[idx].permute(1, 2, 0).cpu().numpy()
        axes[idx, 0].imshow(img)
        axes[idx, 0].set_title(f'Original\n{class_names[test_labels[idx]]}')
        axes[idx, 0].axis('off')
        
        # SHAP values pentru clasa prezisă corect
        shap_img = np.abs(shap_values[test_labels[idx]][idx]).sum(axis=0)
        shap_img = (shap_img - shap_img.min()) / (shap_img.max() - shap_img.min() + 1e-8)
        axes[idx, 1].imshow(img)
        axes[idx, 1].imshow(shap_img, cmap='RdBu_r', alpha=0.6)
        axes[idx, 1].set_title('SHAP')
        axes[idx, 1].axis('off')
    
    plt.suptitle('SHAP Explanations - Contribuția pixelilor', fontsize=14, fontweight='bold')
    plt.tight_layout()
    save_path = os.path.join(Config.VIZ_DIR, 'shap_explanations.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"[XAI] Salvat în: {save_path}")
    #plt.show()
    plt.close()


# ============================================================================
# ARHITECTURI MODELE - MLP, CNN, ResNeXt
# ============================================================================

class MLPClassifier(nn.Module):
    """
    Multi-Layer Perceptron (MLP) - Rețea Neurală Fully-Connected.
    
    Arhitectură simplă care primește imaginea "aplatizată" (flat vector)
    și aplică straturi dense (fully connected). Nu este ideală pentru imagini
    deoarece nu ține cont de structura spațială 2D a pixelilor.
    
    Parametri:
        num_classes: Numărul de clase de ieșire
        input_size: Dimensiunea vectorului de intrare (3*224*224 = 150528)
        dropout_rate: Rata de dropout pentru regularizare (default 0.5)
    """
    
    def __init__(self, num_classes, input_size=3*224*224, dropout_rate=0.5):
        super(MLPClassifier, self).__init__()
        
        # nn.Sequential: aplică straturile în ordine
        self.model = nn.Sequential(
            # Flatten: transformă (C, H, W) în vector plat (C*H*W,)
            nn.Flatten(),
            
            # Strat 1: 150528 -> 512 neuroni
            # - nn.Linear: strat fully-connected (fiecare neuron conectat la toți)
            # - nn.BatchNorm1d: normalizează activările (stabilizează antrenarea)
            # - nn.ReLU: funcție de activare (introduce neliniaritate)
            # - nn.Dropout: oprit aleatoriu neuroni pentru prevenire overfitting
            nn.Linear(input_size, 512), 
            nn.BatchNorm1d(512), 
            nn.ReLU(), 
            nn.Dropout(dropout_rate),
            
            # Strat 2: 512 -> 256 neuroni
            nn.Linear(512, 256), 
            nn.BatchNorm1d(256), 
            nn.ReLU(), 
            nn.Dropout(dropout_rate),
            
            # Strat 3: 256 -> 128 neuroni
            nn.Linear(256, 128), 
            nn.BatchNorm1d(128), 
            nn.ReLU(), 
            nn.Dropout(dropout_rate),
            
            # Strat de ieșire: 128 -> num_classes (fără activare, aplicăm softmax la loss)
            nn.Linear(128, num_classes)
        )
    
    def forward(self, x):
        """
        Forward pass - trece Datele prin rețea.
        
        Parametri:
            x (torch.Tensor): Tensor de intrare cu shape (batch_size, 3, 224, 224)
        """
        return self.model(x)


class CNNClassifier(nn.Module):
    """
    Convolutional Neural Network (CNN) - Rețea Neurală Convolutivă.
    
    Arhitectura specifică pentru imagini care folosește straturi convolutive:
    - Convoluția aplică filtre mici (3x3) care detectează caracteristici locale
    - Pooling reduce dimensiunea spațială (downsampling)
    - Menține relația spațială între pixeli (față de MLP)
    
    Arhitectură:
    - 4 block-uri convolutive (conv -> batch_norm -> relu -> pooling/dropout)
    - 1 bloc clasificator (flatten -> fc -> batch_norm -> relu -> dropout -> fc_out)
    
    Parametri:
        num_classes: Numărul de clase de ieșire
        dropout_rate: Dropout pentru fc layers (default 0.5)
    """
    
    def __init__(self, num_classes, dropout_rate=0.5):
        super(CNNClassifier, self).__init__()
        
        # ---- FEATURE EXTRACTION (straturi convolutive) ----
        # Fiecare block: Conv -> BatchNorm -> ReLU -> MaxPool/Dropout2d
        
        self.features = nn.Sequential(
            # ---- BLOCK 1: 3 canale -> 32 filtre ----
            # Extrage caracteristici de bază: margini, culori, texturi simple
            nn.Conv2d(3, 32, kernel_size=3, padding=1),   # (224,224,3) -> (224,224,32)
            nn.BatchNorm2d(32),                           # Normalizează cele 32 canale
            nn.ReLU(inplace=True),                        # Neliniraritate
            nn.Conv2d(32, 32, kernel_size=3, padding=1),  # (224,224,32) -> (224,224,32)
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),        # (224,224,32) -> (112,112,32)
            nn.Dropout2d(0.25),                           # Dropout 25% pe canale (regularizare)
            
            # ---- BLOCK 2: 32 -> 64 filtre ----
            # Extrage caracteristici mai complexe: colțuri, forme geometrice
            nn.Conv2d(32, 64, kernel_size=3, padding=1),  # (112,112,32) -> (112,112,64)
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),  # (112,112,64) -> (112,112,64)
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),        # (112,112,64) -> (56,56,64)
            nn.Dropout2d(0.25),
            
            # ---- BLOCK 3: 64 -> 128 filtre ----
            # Extrage forme mai avansate: cercuri, dreptunghiuri, pattern-uri
            nn.Conv2d(64, 128, kernel_size=3, padding=1), # (56,56,64) -> (56,56,128)
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),# (56,56,128) -> (56,56,128)
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),        # (56,56,128) -> (28,28,128)
            nn.Dropout2d(0.25),
            
            # ---- BLOCK 4: 128 -> 256 filtre ----
            # Extrage caracteristici foarte abstracte: părți de obiecte
            nn.Conv2d(128, 256, kernel_size=3, padding=1),# (28,28,128) -> (28,28,256)
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),# (28,28,256) -> (28,28,256)
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),        # (28,28,256) -> (14,14,256)
            nn.Dropout2d(0.25)
        )
        
        # ---- CLASSIFIER (straturi fully-connected) ----
        # Flatten -> Linear -> ReLU -> Dropout -> Linear(output)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            # 256 * 14 * 14 = 50176 este dimensiunea de intrare
            # (14x14 este dimensiunea feature map-ului după 4 poolings: 224->112->56->28->14)
            nn.Linear(256 * 14 * 14, 512),  # Reducere: 50176 -> 512
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),  # Dropout 50% - regularizare puternică
            nn.Linear(512, num_classes)  # Strat de ieșire: 512 -> num_classes
        )
    
    def forward(self, x):
        """
        Forward pass - trece datele prin rețea.
        
        Parametri:
            x (torch.Tensor): Tensor de intrare cu shape (batch_size, 3, 224, 224)
        
        Returnări:
            torch.Tensor: Logits de ieșire cu shape (batch_size, num_classes)
        """
        # Mai întâi extragem caracteristicile cu straturile convolutive
        x = self.features(x)
        # Apoi clasificăm cu straturile fully-connected
        x = self.classifier(x)
        return x


class ResNeXtClassifier(nn.Module):
    """
    ResNeXt-50 cu Transfer Learning - Arhitectură avansată pre-antrenată.
    
    ResNeXt este o extensie a ResNet care folosește "cardinalitatea" (numărul
    de căi paralele în loc de adâncimea) pentru reprezentări mai bogate.
    
    Transfer Learning:
    - Încărcăm ponderile pre-antrenate pe ImageNet (1.2M imagini, 1000 clase)
    - Modelul a învă deja caracteristici generale: margini, texturi, forme
    - Înlocuim ultimul strat (fc) cu unul nou pentru clasele noastre
    - Înghețăm backbone-ul (nu antrenăm) - doar antrenăm noul fc
    - După epoca 10, dezghețăm backbone-ul pentru fine-tuning cu LR mic
    
    De ce ResNeXt?
    - Performanță excelentă pe ImageNet (top 5 eroare < 5%)
    - Arhitectură robustă cu skip connections (previne vanishing gradient)
    - Cardinalitatea (32 de căi) oferă reprezentări mai bogate decât ResNet
    
    Parametri:
        num_classes: Numărul de clase de ieșire
        pretrained (bool): Dacă True, încarcă ponderile ImageNet
        freeze_backbone (bool): Dacă True, îngheață straturile convolutionale
    """
    
    def __init__(self, num_classes, pretrained=True, freeze_backbone=True):
        super(ResNeXtClassifier, self).__init__()
        
        # Încărcăm modelul pre-antrenat cu_weights
        # IMAGENET1K_V1 = antrenat pe ImageNet (1000 clase)
        # Aceste ponderile conțin caracteristici generale utile pentru orice task
        weights = models.ResNeXt50_32X4D_Weights.IMAGENET1K_V1 if pretrained else None
        self.backbone = models.resnext50_32x4d(weights=weights)
        
        # Dimensiunea de intrare a stratului fully-connected original
        # ResNeXt-50 are ultimul layer cu 2048 de intrări
        num_features = self.backbone.fc.in_features
        
        # Înghețăm toate ponderile din backbone
        # Astfel, doar noul strat fc va fi antrenat inițial
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False  # Nu calculăm gradient pentru aceste ponderi
        
        # Înlocuim ultimul strat (fc) cu unul nou pentru clasele noastre
        # Vechiul strat: 2048 -> 1000 (ImageNet classes)
        # Noul strat: 2048 -> 512 -> num_classes (classesle noastre)
        self.backbone.fc = nn.Sequential(
            nn.Dropout(0.5),                    # Dropout 50% regularizare
            nn.Linear(num_features, 512),       # Reducere: 2048 -> 512
            nn.ReLU(inplace=True),              # Nelinieraritate
            nn.Dropout(0.3),                    # Dropout 30% suplimentar
            nn.Linear(512, num_classes)         # Ieșire: 512 -> num_classes
        )
        # Inițializarea automată a noilor ponderi (PyTorch o face default)
    
    def forward(self, x):
        """
        Forward pass - trece datele prin ResNeXt.
        
        Parametri:
            x (torch.Tensor): (batch_size, 3, 224, 224)
        """
        return self.backbone(x)
    
    def unfreeze_backbone(self):
        """
        Dezgheață backbone-ul pentru fine-tuning.
        
        Se apelează după câteva epoci când noul fc s-a antrenat suficient
        și vrem să ajustăm ușor și caracteristicile convolutionale.
        """
        for param in self.backbone.parameters():
            param.requires_grad = True


# ============================================================================
# ANTRENARE ȘI VALIDARE - Epoch, Training, Evaluation
# ============================================================================

def train_epoch(model, train_loader, criterion, optimizer, device, epoch, writer=None):
    """
    Antrenează modelul pentru o epocă completă.
    
    O epocă = o trecere completă prin toate datele de antrenare.
    
    Parametri:
        model: Modelul neural de antrenat
        train_loader: DataLoader pentru datele de antrenare
        criterion: Funcția de loss (CrossEntropyLoss)
        optimizer: Optimizatorul (Adam)
        device: CPU sau GPU
        epoch: Numărul epocii curente (pentru logging)
        writer: TensorBoard SummaryWriter (opțional)
    
    Returnări:
        tuple: (average_loss, accuracy) pentru această epocă
    """
    model.train()  # Setează modul de antrenare (activează Dropout, BatchNorm update)
    running_loss = 0.0
    correct = 0
    total = 0
    
    # tqdm - bară de progres
    pbar = tqdm(train_loader, desc=f"Epoca {epoch+1} [Antrenare]", total=len(train_loader), ncols=100)
    
    for images, labels in pbar:
        # Transferăm datele pe GPU (dacă e disponibil)
        # non_blocking=True permite transferul async pentru viteză mai mare
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        
        # Zero gradients - resetăm gradientul înainte de fiecare forward pass
        # Gradientul se acumulează, deci trebuie resetat manual
        optimizer.zero_grad()
        
        # Forward pass - predicția modelului
        outputs = model(images)
        
        # Calculăm loss-ul (diferența dintre predicții și etichetele reale)
        # CrossEntropyLoss include softmax, deci nu aplicăm noi
        loss = criterion(outputs, labels)
        
        # Backward pass - calculăm gradientul pentru fiecare ponder
        loss.backward()
        
        # Update ponderi - optimizer aplică actualizările
        # lr * gradient = cât de mult ajustăm fiecare ponder
        optimizer.step()
        
        # Statistici
        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)  # Clasa cu scor maxim
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
        # Actualizăm bara de progres
        pbar.set_postfix({'Loss': f'{loss.item():.4f}', 'Acc': f'{100.0 * correct / total:.2f}%'})
    
    # Medii pe întreaga epocă
    epoch_loss = running_loss / len(train_loader)
    epoch_acc = 100.0 * correct / total
    
    # Logging TensorBoard
    if writer:
        writer.add_scalars('Loss', {'train': epoch_loss}, epoch)
        writer.add_scalars('Accuracy', {'train': epoch_acc}, epoch)
    
    return epoch_loss, epoch_acc


def validate(model, val_loader, criterion, device, epoch, writer=None):
    """
    Evaluează modelul pe datele de validare.
    
    Diferențe față de train_epoch:
    - model.eval() - dezactivează Dropout, folosește statistici BatchNorm fixe
    - torch.no_grad() - nu calculăm gradient (economisim memorie și timp)
    - Nu actualizăm ponderile (nici optimizer.step())
    
    Returnări:
        tuple: (average_loss, accuracy) pe validare
    """
    model.eval()  # Mod de evaluare (dezactivează Dropout)
    running_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(val_loader, desc=f"Epoca {epoch+1} [Validare]", total=len(val_loader), ncols=100)
    
    # Context manager fără gradient - mai rapid și mai puțină memorie
    with torch.no_grad():
        # Nu actualizăm ponderile, doar evaluăm
        for images, labels in pbar:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            pbar.set_postfix({'Loss': f'{loss.item():.4f}', 'Acc': f'{100.0 * correct / total:.2f}%'})
    
    epoch_loss = running_loss / len(val_loader)
    epoch_acc = 100.0 * correct / total
    
    if writer:
        writer.add_scalars('Loss', {'val': epoch_loss}, epoch)
        writer.add_scalars('Accuracy', {'val': epoch_acc}, epoch)
    
    return epoch_loss, epoch_acc


def train_model(model, model_name, train_loader, val_loader, num_classes, device, run_timestamp, run_name=None):
    """
    Antrenează un model complet cu LR Scheduler și Early Stopping,
    logging în MLflow pentru comparare între experimente.
    
    MLflow:
    - Fiecare rulare este un "run" separat în MLflow
    - Putem compara rulări diferite prin interfața MLflow UI
    - Toți hiperparametrii sunt salvați pentru fiecare run
    - Metricile (loss, acc) sunt logate la fiecare epocă
    
    Parametri:
        model: Modelul neural de antrenat
        model_name: Numele arhitecturii (MLP/CNN/ResNeXt)
        train_loader: DataLoader antrenare
        val_loader: DataLoader validare
        num_classes: Numărul de clase
        device: CPU sau GPU
        run_timestamp: Timestamp pentru salvare consistentă
        run_name: Numele run-ului în MLflow (opțional, default=model_name)
        
    Returnări:
        tuple: (history_dict, trained_model)
    """
    print(f"\n{'='*60}")
    print(f"ANTRENARE: {model_name}")
    if run_name:
        print(f"RUN NAME: {run_name}")
    print(f"{'='*60}")
    
    model = model.to(device)
    
    # Numărăm parametrii modelului
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[INFO] Total parametri: {total_params:,}")
    print(f"[INFO] Parametri antrenabili: {trainable_params:,}")
    
    # ---- Loss Function ----
    # CrossEntropyLoss = Softmax + CrossEntropy
    # Este standard pentru clasificare multi-clasă
    # Combinația: aplică log-softmax și apoi negative log likelihood loss
    criterion = nn.CrossEntropyLoss()
    
    # ---- Optimizer ----
    # Adam = Adaptive Moment Estimation
    # Combină avantajurile lui SGD cu momentum și RMSprop
    # - lr=0.001: learning rate standard pentru Adam
    # - weight_decay=1e-4: regularizare L2 (previne overfitting)
    # filter(lambda p: p.requires_grad, ...) - optimizăm doar ponderile care necesită gradient
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=Config.LEARNING_RATE,
        weight_decay=Config.WEIGHT_DECAY
    )
    
    # ---- LR Scheduler ----
    # ReduceLROnPlateau: scade learning rate când loss-ul pe validare plateau
    # - mode='min': monitorizăm minimul loss-ului
    # - factor=0.5: când plateau, lr = lr * 0.5 (înjumătățim)
    # - patience=3: așteptăm 3 epoci fără îmbunătățire înainte de reducere
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3
    )
    
    # ---- Early Stopping ----
    best_val_loss = float('inf')
    epochs_without_improvement = 0
    
    # ---- TensorBoard Logger ----
    log_dir = os.path.join(Config.LOG_DIR, f"{model_name}_{run_timestamp}")
    writer = SummaryWriter(log_dir)

    # Generează un tensor dummy cu dimensiunile (Batch, Canale, Înălțime, Lățime)
    dummy_input = torch.randn(1, 3, Config.IMG_SIZE, Config.IMG_SIZE).to(device)
    writer.add_graph(model, dummy_input)
    
    # ---- MLflow Setup ----
    # Setăm experimentul MLflow
    mlflow.set_experiment(Config.MLFLOW_EXPERIMENT_NAME)
    
    # Istoric pentru grafice
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }
    
    best_val_acc = 0.0
    checkpoint_path = f"best_{model_name}_{run_timestamp}.pth"
    
    # ---- MLflow Run Context ----
    # Fiecare model primește un run_name unic pentru comparare în MLflow UI
    if run_name is None:
        run_name = model_name
    
    with mlflow.start_run(run_name=run_name) as run:
        # Logăm TOȚI hiperparametrii în MLflow
        mlflow.log_params({
            "model_name": model_name,
            "run_name": run_name,
            "num_classes": num_classes,
            "batch_size": Config.BATCH_SIZE,
            "learning_rate": Config.LEARNING_RATE,
            "weight_decay": Config.WEIGHT_DECAY,
            "dropout_rate": Config.DROPOUT_RATE,
            "epochs": Config.EPOCHS,
            "early_stopping_patience": Config.EARLY_STOPPING_PATIENCE,
            "early_stopping_min_delta": Config.EARLY_STOPPING_MIN_DELTA,
            "occlusion_prob": Config.OCCLUSION_PROB,
            "image_size": Config.IMG_SIZE,
            "train_split_ratio": Config.TRAIN_SPLIT,
            "total_params": total_params,
            "trainable_params": trainable_params,
            "optimizer": "Adam",
            "loss_function": "CrossEntropyLoss",
            "lr_scheduler": "ReduceLROnPlateau",
            "lr_scheduler_factor": 0.5,
            "lr_scheduler_patience": 3,
            "run_timestamp": run_timestamp
        })
        
        # Logăm și arhitectura modelului
        print(f"[MLflow] Run ID: {run.info.run_id}")
        
        for epoch in range(Config.EPOCHS):
            print(f"\n[EPOCA {epoch+1}/{Config.EPOCHS}]")
            print("-" * 60)
            
            start_time = time.time()
            
            # Antrenare
            train_loss, train_acc = train_epoch(
                model, train_loader, criterion, optimizer, device, epoch, writer
            )
            
            # Validare
            val_loss, val_acc = validate(
                model, val_loader, criterion, device, epoch, writer
            )
            
            epoch_time = time.time() - start_time
            
            # Salvăm în istoric
            history['train_loss'].append(train_loss)
            history['train_acc'].append(train_acc)
            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_acc)
            
            print(f"\n[REZULTATE EPOCA {epoch+1}]")
            print(f"  Timp: {epoch_time:.2f}s")
            print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
            print(f"  Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.2f}%")
            
            # MLflow logging per epocă
            mlflow.log_metrics({
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_loss": val_loss,
                "val_acc": val_acc,
                "epoch_time": epoch_time,
                "learning_rate": optimizer.param_groups[0]['lr']
            }, step=epoch)
            
            # LR Scheduler - actualizăm după validare
            prev_lr = optimizer.param_groups[0]['lr']
            scheduler.step(val_loss)  # Scade LR dacă loss-ul nu se îmbunătățește
            new_lr = optimizer.param_groups[0]['lr']
            if new_lr != prev_lr:
                print(f"  >> LR schimbat: {prev_lr:.6f} -> {new_lr:.6f}")
                mlflow.log_metric("learning_rate_change", new_lr, step=epoch)
            
            # ---- Early Stopping Check ----
            if val_loss < best_val_loss - Config.EARLY_STOPPING_MIN_DELTA:
                # Am îmbunătățire - resetăm counter-ul
                best_val_loss = val_loss
                epochs_without_improvement = 0
            else:
                # Nu avem îmbunătățire
                epochs_without_improvement += 1
                print(f"  >> Early stopping: {epochs_without_improvement}/{Config.EARLY_STOPPING_PATIENCE}")
                
                if epochs_without_improvement >= Config.EARLY_STOPPING_PATIENCE:
                    print(f"\n{'='*60}")
                    print(f"[INFO] Early stopping activat la epoca {epoch+1}")
                    print(f"{'='*60}")
                    break
            
            # ---- Salvați cel mai bun model ----
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_acc': val_acc,
                    'val_loss': val_loss,
                    'run_timestamp': run_timestamp,
                    'model_name': model_name,
                    'run_name': run_name
                }, checkpoint_path)
                print(f"  >> Model salvat! (Val Acc: {val_acc:.2f}%)")
            
            # ---- Fine-tuning ResNeXt la epoca 10 ----
            # Dezghețăm backbone-ul și continuăm cu LR mai mic
            if model_name == "ResNeXt" and epoch == 10:
                print("[INFO] Dezghețare backbone pentru fine-tuning...")
                if hasattr(model, 'unfreeze_backbone'):
                    model.unfreeze_backbone()
                    # Count new trainable params
                    new_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
                    print(f"[INFO] Parametri antrenabili acum: {new_trainable:,}")
                    mlflow.log_param("trainable_params_after_unfreeze", new_trainable)
                
                # LR mai mic pentru fine-tuning (nu vrem să distrugem ponderile pre-antrenate)
                optimizer = optim.Adam(
                    model.parameters(),
                    lr=Config.LEARNING_RATE / 10,  # 0.0001 în loc de 0.001
                    weight_decay=Config.WEIGHT_DECAY
                )
                print(f"[INFO] Learning rate nou: {Config.LEARNING_RATE / 10}")
                mlflow.log_metric("fine_tuning_lr", Config.LEARNING_RATE / 10, step=epoch)
        
        # ---- Final MLflow Logging ----
        # Logăm modelul final în MLflow
        mlflow.log_metric("best_val_acc", best_val_acc)
        mlflow.log_metric("best_val_loss", best_val_loss)
        mlflow.log_metric("final_train_acc", history['train_acc'][-1])
        mlflow.log_metric("final_train_loss", history['train_loss'][-1])
        mlflow.log_metric("total_epochs_trained", len(history['train_acc']))
        
        # Logăm checkpoint-ul ca artifact
        if os.path.exists(checkpoint_path):
            mlflow.log_artifact(checkpoint_path)
        
        # Logăm modelul PyTorch
        mlflow.pytorch.log_model(model, "model")
        
        # Logăm graficele de training ca artifact
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        
        # Grafic Loss
        axes[0].plot(history['train_loss'], label='Train Loss', marker='o')
        axes[0].plot(history['val_loss'], label='Val Loss', marker='s')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].set_title('Training & Validation Loss')
        axes[0].legend()
        axes[0].grid(True)
        
        # Grafic Accuracy
        axes[1].plot(history['train_acc'], label='Train Acc', marker='o')
        axes[1].plot(history['val_acc'], label='Val Acc', marker='s')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Accuracy (%)')
        axes[1].set_title('Training & Validation Accuracy')
        axes[1].legend()
        axes[1].grid(True)
        
        plt.tight_layout()
        viz_path = os.path.join(Config.VIZ_DIR, f'training_curves_{model_name}_{run_timestamp}.png')
        os.makedirs(Config.VIZ_DIR, exist_ok=True)
        plt.savefig(viz_path, dpi=150, bbox_inches='tight')
        mlflow.log_artifact(viz_path)
        plt.close()
        
        print(f"\n{'='*60}")
        print(f"ANTRENARE FINALIZATĂ: {model_name}")
        print(f"Cea mai bună acuratețe: {best_val_acc:.2f}%")
        print(f"Model salvat: {checkpoint_path}")
        print(f"MLflow Run ID: {run.info.run_id}")
        print(f"{'='*60}")
    
    writer.close()
    return history, model


# ============================================================================
# FUNCȚIA PRINCIPALĂ
# ============================================================================

def compare_transforms():
    """
    Antrenează CNN de două ori — fără și cu RandomGrayscale + GaussianBlur —
    și compară rezultatele. Fiecare rulare e logată separat în MLflow.

    Returnări:
        dict: {'without': best_val_acc, 'with': best_val_acc}
    """
    print("=" * 70)
    print("COMPARARE TRANSFORMĂRI: fără vs. cu RandomGrayscale + GaussianBlur")
    print("=" * 70)

    set_seed(42)
    results = {}

    for use_extra, label in [(False, "fara_extra_transforms"), (True, "cu_extra_transforms")]:
        Config.USE_EXTRA_TRANSFORMS = use_extra
        tag = "CU" if use_extra else "FARA"
        print(f"\n[{tag}] RandomGrayscale + GaussianBlur")
        print("-" * 60)

        train_loader, val_loader, num_classes, class_names = load_data()
        model = CNNClassifier(num_classes=num_classes)
        run_timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        history, _ = train_model(
            model=model,
            model_name='CNN',
            train_loader=train_loader,
            val_loader=val_loader,
            num_classes=num_classes,
            device=Config.DEVICE,
            run_timestamp=run_timestamp,
            run_name=f"CNN_{label}",
        )

        best_acc = max(history['val_acc'])
        best_loss = min(history['val_loss'])
        results[label] = {'val_acc': best_acc, 'val_loss': best_loss}
        print(f"  >> Best Val Acc: {best_acc:.2f}%  |  Best Val Loss: {best_loss:.4f}")

    # ---- Tabel comparativ ----
    print("\n" + "=" * 70)
    print("REZULTATE COMPARARE TRANSFORMĂRI")
    print("=" * 70)
    print(f"{'Configurație':<35} {'Val Acc':>8} {'Val Loss':>10}")
    print("-" * 55)
    for label, r in results.items():
        print(f"{label:<35} {r['val_acc']:>7.2f}%  {r['val_loss']:>9.4f}")
    print("=" * 70)

    acc_fara = results['fara_extra_transforms']['val_acc']
    acc_cu   = results['cu_extra_transforms']['val_acc']
    diff = acc_cu - acc_fara

    if diff > 0:
        print(f"\n[CONCLUZIE] Transformările suplimentare AJUTĂ: +{diff:.2f}% acuratețe")
    elif diff < 0:
        print(f"\n[CONCLUZIE] Transformările suplimentare SCAD acuratețea: {diff:.2f}%")
        print("  (posibil prea multă augmentare pentru dataset-ul mic de 360 imagini)")
    else:
        print("\n[CONCLUZIE] Nicio diferență semnificativă.")

    print("[MLflow] Compară cele 2 rulări la: mlflow ui  ->  http://localhost:5000")
    return results


def run_hyperparameter_search():
    """
    Rulează un grid search peste combinații de BATCH_SIZE, LEARNING_RATE și EPOCHS.

    Fiecare combinație este antrenată pe modelul Config.SEARCH_MODEL și logată
    în MLflow cu un run_name descriptiv (ex: CNN_BS32_LR0.0100_E30).
    La final afișează un tabel sortat după best_val_acc.

    Returnări:
        list[dict]: Rezultatele sortate descrescător după acuratețe validare.
    """
    print("=" * 70)
    print("GRID SEARCH HIPERPARAMETRI")
    print("=" * 70)
    print(f"Model: {Config.SEARCH_MODEL}")
    print(f"BATCH_SIZE:     {Config.SEARCH_BATCH_SIZES}")
    print(f"LEARNING_RATE:  {Config.SEARCH_LEARNING_RATES}")
    print(f"EPOCHS:         {Config.SEARCH_EPOCHS}")
    total_runs = (len(Config.SEARCH_BATCH_SIZES) *
                  len(Config.SEARCH_LEARNING_RATES) *
                  len(Config.SEARCH_EPOCHS))
    print(f"Total rulări:   {total_runs}")
    print("=" * 70)

    set_seed(42)

    # Determinăm num_classes o singură dată (nu depinde de hiperparametri)
    _tmp_dataset = __import__('torchvision').datasets.ImageFolder(root=Config.DATA_DIR)
    if Config.SELECTED_CLASSES:
        num_classes = len(Config.SELECTED_CLASSES)
        class_names = Config.SELECTED_CLASSES
    else:
        num_classes = len(_tmp_dataset.classes)
        class_names = _tmp_dataset.classes

    model_registry = {
        'MLP': MLPClassifier,
        'CNN': CNNClassifier,
        'ResNeXt': lambda nc: ResNeXtClassifier(num_classes=nc, pretrained=True, freeze_backbone=True),
    }
    ModelClass = model_registry[Config.SEARCH_MODEL]

    results_summary = []
    run_idx = 0

    for bs in Config.SEARCH_BATCH_SIZES:
        for lr in Config.SEARCH_LEARNING_RATES:
            for epochs in Config.SEARCH_EPOCHS:
                run_idx += 1
                # Actualizăm Config în loc să transmitem parametri individuali
                Config.BATCH_SIZE = bs
                Config.LEARNING_RATE = lr
                Config.EPOCHS = epochs

                run_name = f"{Config.SEARCH_MODEL}_BS{bs}_LR{lr:.4f}_E{epochs}"
                print(f"\n[{run_idx}/{total_runs}] {run_name}")
                print("-" * 60)

                # Reîncărcăm datele cu noul BATCH_SIZE
                train_loader, val_loader, _, _ = load_data()

                model = ModelClass(num_classes) if Config.SEARCH_MODEL != 'ResNeXt' else ModelClass(num_classes)

                run_timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                history, _ = train_model(
                    model=model,
                    model_name=Config.SEARCH_MODEL,
                    train_loader=train_loader,
                    val_loader=val_loader,
                    num_classes=num_classes,
                    device=Config.DEVICE,
                    run_timestamp=run_timestamp,
                    run_name=run_name,
                )

                best_acc = max(history['val_acc'])
                best_loss = min(history['val_loss'])
                epochs_trained = len(history['val_acc'])

                results_summary.append({
                    'run_name': run_name,
                    'batch_size': bs,
                    'learning_rate': lr,
                    'epochs': epochs,
                    'epochs_trained': epochs_trained,
                    'best_val_acc': best_acc,
                    'best_val_loss': best_loss,
                })
                print(f"  >> Best Val Acc: {best_acc:.2f}%  |  Epoci rulate: {epochs_trained}")

    # ---- Tabel rezultate ----
    results_summary.sort(key=lambda x: x['best_val_acc'], reverse=True)

    print("\n" + "=" * 70)
    print("REZULTATE GRID SEARCH — sortat după Val Acc (descrescător)")
    print("=" * 70)
    header = f"{'#':>3}  {'Run Name':<38} {'BS':>4} {'LR':>8} {'EP':>4} {'EPr':>4} {'Val Acc':>8} {'Val Loss':>9}"
    print(header)
    print("-" * 70)
    for i, r in enumerate(results_summary, 1):
        print(
            f"{i:>3}  {r['run_name']:<38} {r['batch_size']:>4} "
            f"{r['learning_rate']:>8.4f} {r['epochs']:>4} {r['epochs_trained']:>4} "
            f"{r['best_val_acc']:>7.2f}%  {r['best_val_loss']:>8.4f}"
        )
    print("=" * 70)

    best = results_summary[0]
    print(f"\n[WINNER] {best['run_name']}")
    print(f"  BATCH_SIZE    = {best['batch_size']}")
    print(f"  LEARNING_RATE = {best['learning_rate']}")
    print(f"  EPOCHS        = {best['epochs']}")
    print(f"  Val Acc       = {best['best_val_acc']:.2f}%")
    print(f"  Val Loss      = {best['best_val_loss']:.4f}")
    print("\n[MLflow] Compară toate rulările la: mlflow ui  ->  http://localhost:5000")

    return results_summary


def main():
    """
    Funcția principală care rulează întregul pipeline.
    
    Fluxul este:
    1. Configurare și seed
    2. Încărcare date
    3. Vizualizare transformări
    4. Antrenare fiecare model
    5. Vizualizări XAI pe cel mai bun model
    6. Afișare rezultate finale
    """
    print("=" * 70)
    print("CLASIFICATOR DE FRUCTE CU PYTORCH")
    print("=" * 70)
    print("\nCaracteristici:")
    print("  - 5 clase (extensibile la 9)")
    print("  - Data Augmentation: Flip, Rotate, Zoom, Grayscale, Blur")
    print("  - Random Pixel Removal pentru ocluzie")
    print("  - Learning Rate Scheduler și Early Stopping")
    print("  - Urmărire experimente cu MLflow")
    print("=" * 70)
    
    # Setăm seed pentru reproducibilitate
    set_seed(42)
    Config.print_config()

    # ---- Mod Grid Search ----
    # Dacă HYPERPARAMETER_SEARCH = True, rulăm grid search și ieșim
    if Config.HYPERPARAMETER_SEARCH:
        run_hyperparameter_search()
        return

    # ---- Mod Comparare Transformări ----
    # Dacă COMPARE_TRANSFORMS = True, antrenăm CNN cu/fără transformări extra
    if Config.COMPARE_TRANSFORMS:
        compare_transforms()
        return

    # Timestamp unic pentru această rulare
    # Toate modelele din aceeași rulare vor avea același timestamp
    run_timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    print(f"\n[INFO] Timestamp rulare: {run_timestamp}")
    print("[INFO] Toate modelele vor fi salvate cu acest timestamp\n")
    
    # ---- Încărcare date ----
    train_loader, val_loader, num_classes, class_names = load_data()
    
    # ---- Vizualizare transformări ----
    print("\n[VIZ] Generare vizualizări...")
    visualize_transforms(num_images=6)
    
    # ---- Creare modele ----
    # Definim cele 3 arhitecturi de testat
    models_dict = {
        'MLP': MLPClassifier(num_classes=num_classes),
        'CNN': CNNClassifier(num_classes=num_classes),
        'ResNeXt': ResNeXtClassifier(num_classes=num_classes, pretrained=True, freeze_backbone=True)
    }
    
    # ---- Antrenare fiecare model ----
    results = {}
    trained_models = {}
    run_names = {}  # Track each model's MLflow run name
    
    for model_name, model in models_dict.items():
        print(f"\n{'#'*70}")
        print(f"# PROCESARE MODEL: {model_name}")
        print(f"{'#'*70}")
        
        # Creăm un run_name unic pentru MLflow
        # Format: "model_name_iteration" pentru a putea compara rulări
        run_name = f"{model_name}"#_{run_timestamp}"
        run_names[model_name] = run_name

        # Dacă vrem să includem hiperparametrii în run_name pentru comparare mai detaliată, putem face asta aici
        # run_name = f"{model_name}_LR{Config.LEARNING_RATE}_BS{Config.BATCH_SIZE}"
        # run_names[model_name] = run_name
        
        # Antrenăm modelul
        history, trained_model = train_model(
            model=model,
            model_name=model_name,
            train_loader=train_loader,
            val_loader=val_loader,
            num_classes=num_classes,
            device=Config.DEVICE,
            run_timestamp=run_timestamp,
            run_name=run_name
        )
        
        results[model_name] = {'history': history}
        trained_models[model_name] = trained_model
        
        print(f"\n[MLflow] Experiment '{model_name}' salvat cu run_name: '{run_name}'")
    
    # ---- Vizualizări XAI ----
    print("\n" + "=" * 70)
    print("GENERARE VIZUALIZĂRI XAI")
    print("=" * 70)
    
    # Folosim cel mai bun model (ResNeXt de obicei) pentru XAI
    best_model_name = 'ResNeXt' if 'ResNeXt' in trained_models else list(trained_models.keys())[-1]
    best_model = trained_models[best_model_name]
    
    print(f"\n[INFO] Folosind modelul {best_model_name} pentru XAI")
    
    print("\n[1/3] Analiza predicțiilor...")
    analyze_predictions(best_model, val_loader, class_names, Config.DEVICE)
    
    print("\n[2/3] Generare Saliency Maps...")
    visualize_saliency_maps(best_model, val_loader, class_names, Config.DEVICE)
    
    print("\n[3/3] Generare SHAP...")
    explain_with_shap(best_model, val_loader, class_names, Config.DEVICE)
    
    # ---- Rezultate finale ----
    print("\n" + "=" * 70)
    print("REZULTATE FINALE")
    print("=" * 70)
    for model_name, result in results.items():
        best_val_acc = max(result['history']['val_acc'])
        checkpoint_file = f"best_{model_name}_{run_timestamp}.pth"
        print(f"{model_name:10s}: {best_val_acc:.2f}% | {checkpoint_file}")
    print("=" * 70)
    
    print(f"\n[INFO] Vizualizări salvate în: {Config.VIZ_DIR}/")
    print("[INFO] TensorBoard: tensorboard --logdir=runs/fruit_classifier")
    print("[INFO] MLflow: mlflow ui")
    print("[INFO] Compară experimente în MLflow UI: http://localhost:5000")


if __name__ == "__main__":
    main()
