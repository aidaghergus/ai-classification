# Documentație Clasificator Fructe

## Partea 1: Utilizare

### 1.1 Rulare Aplicație

```bash
# Activează mediul virtual (dacă există)
.venv\Scripts\activate  # Windows
source .venv/bin/activate  # Linux/Mac

# Rulează antrenarea
python main.py
```

**Ce se întâmplă:**
1. Se încarcă datele (5 clase: mere, banane, cireșe, struguri, portocale)
2. Se aplică automat toate transformările (augmentare + occludere)
3. Se antrenează 3 modele: MLP, CNN, ResNeXt
4. Se aplică **Early Stopping** și **Learning Rate Scheduler** automat
5. Se generează vizualizări (transformări, predicții, Saliency Maps, SHAP)
6. Se salvează rezultatele în `visualizations/` și checkpointuri în `best_*_timestamp.pth`

### 1.2 TensorBoard - Vizualizare Live

**În timpul antrenării**, deschide un terminal nou:

```bash
tensorboard --logdir=runs/fruit_classifier
```

Deschide browser: http://localhost:6006

**Grafice disponibile:**
- **Loss**: Comparație train vs validation (selectabil)
- **Accuracy**: Comparație train vs validation (selectabil)
- **Model Graph**: Arhitectura rețelei

**Tips:**
- Bifează/debifează în legendă pentru a arăta doar train sau validation
- Folosește slider-ul de timp pentru a vedea evoluția
- Compară modele diferite (MLP vs CNN vs ResNeXt)

### 1.3 MLflow - Tracking Experimente

```bash
mlflow ui
```

Deschide browser: http://localhost:5000

**Vezi:**
- Toate rulările (runs) cu parametri și metrici
- Comparație între modele
- Artefacte (modele salvate)
- Log-uri Early Stopping și LR Scheduler

---

## Partea 2: Documentație Tehnică

### 2.1 Arhitectura Proiectului

```
main.py
├── Config                          # Configurare hiperparametri + Early Stopping
├── RandomPixelRemoval              # Transformare custom occludere
├── get_transforms()                # Pipeline Data Augmentation
├── FilteredDataset                 # Filtrare și remapare etichete
├── load_data()                     # Încărcare și split date
├── MLPClassifier                   # Model fully-connected (nn.Sequential)
├── CNNClassifier                   # Model convoluțional (nn.Sequential)
├── ResNeXtClassifier               # Transfer learning
├── train_epoch()                   # Loop antrenare o epocă
├── validate()                      # Evaluare
├── train_model()                   # Loop complet cu LR Scheduler + Early Stopping
└── main()                          # Pipeline principal
```

### 2.2 Flow-ul Datelor

```
Imagini (3×224×224)
       ↓
[Data Augmentation] - Flip, Rotate, ColorJitter, Occludere
       ↓
[Normalizare ImageNet] - mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
       ↓
[Model] - MLP / CNN / ResNeXt
       ↓
[LR Scheduler] - Reduce LR când loss stagnează
       ↓
[Early Stopping] - Oprește dacă nu mai există îmbunătățiri
       ↓
[Logare] - TensorBoard + MLflow + Checkpointuri cu timestamp
```

### 2.3 Modele - Detalii Tehnice

#### MLP (Multi-Layer Perceptron)
```
Input: 150,528 (3×224×224 flatten)
  ↓
Layer1: 512 neuroni + BatchNorm + ReLU + Dropout(0.5)
  ↓
Layer2: 256 neuroni + BatchNorm + ReLU + Dropout(0.5)
  ↓
Layer3: 128 neuroni + BatchNorm + ReLU + Dropout(0.5)
  ↓
Output: 5 (număr clase)
```
**Parametri**: ~77M (multe datorită flatten)

#### CNN (Convolutional Neural Network)
```
Block1: Conv(3→32) → BatchNorm → ReLU → Conv(32→32) → Pool → Dropout2d(0.25)
       224×224 → 112×112
Block2: Conv(32→64) → BatchNorm → ReLU → Conv(64→64) → Pool → Dropout2d(0.25)
       112×112 → 56×56
Block3: Conv(64→128) → BatchNorm → ReLU → Conv(128→128) → Pool → Dropout2d(0.25)
       56×56 → 28×28
Block4: Conv(128→256) → BatchNorm → ReLU → Conv(256→256) → Pool → Dropout2d(0.25)
       28×28 → 14×14
Classifier: Flatten(50,176) → Linear(512) → BatchNorm → ReLU → Dropout(0.5) → Output(5)
```
**Parametri**: ~28M (mai eficient decât MLP)

#### ResNeXt (Transfer Learning)
```
ResNeXt-50 (pre-antrenat pe ImageNet)
  ↓
[Congelat] - primele layere înghețate
  ↓
[Custom Classifier] - Dropout(0.5) → Linear(512) → ReLU → Dropout(0.3) → Output(5)
```
**Parametri**: ~25M (cei mai mulți înghețați inițial)

### 2.4 Funcții Cheie

#### `FilteredDataset` - Problema Etichetelor
```python
# PROBLEMĂ: Când filtrăm 5 clase din 9, etichetele rămân [0,1,2,4,6,7]
# dar modelul are 5 output-uri și așteaptă [0,1,2,3,4]

# SOLUȚIE: Remapăm etichetele
class_to_idx = {
    'apple fruit': 0,
    'banana fruit': 1,
    'cherry fruit': 2,
    'grapes fruit': 3,
    'orange fruit': 4
}
```

#### `get_transforms()` - Pipeline Augmentare
```python
# Ordinea transformărilor e importantă:
1. Resize(224,224)           # Dimensiune standard
2. Geometric augmentations   # Flip, Rotate, Perspective
3. Color augmentations       # ColorJitter
4. ToTensor()                # PIL → Tensor
5. Normalize(ImageNet)       # Standardizare
6. RandomPixelRemoval(0.1)   # Occludere artificială
```

#### `train_model()` - Antrenare cu Scheduling și Early Stopping
```python
# 1. Creare optimizer Adam
optimizer = optim.Adam(..., lr=0.001)

# 2. Learning Rate Scheduler
scheduler = ReduceLROnPlateau(
    optimizer, 
    mode='min',         # Urmărește minimizarea loss
    factor=0.5,         # Reduce LR la jumătate
    patience=3,         # Așteaptă 3 epoci
    verbose=True        # Afișează schimbările
)

# 3. Early Stopping
best_val_loss = float('inf')
epochs_no_improve = 0

for epoch in range(EPOCHS):
    train_loss, val_loss = ...
    
    # Actualizare scheduler
    scheduler.step(val_loss)
    
    # Verificare Early Stopping
    if val_loss < best_val_loss - MIN_DELTA:
        best_val_loss = val_loss
        epochs_no_improve = 0
    else:
        epochs_no_improve += 1
        if epochs_no_improve >= PATIENCE:
            break  # Oprește antrenarea
```

### 2.5 Loss, Optimizare și Scheduling

**CrossEntropyLoss**
- Combinație între LogSoftmax și Negative Log Likelihood
- Standard pentru clasificare multi-clasă

**Adam Optimizer**
- Learning rate: 0.001
- Weight decay: 1e-4 (L2 regularization)
- Adaptive learning rates per parameter

**Learning Rate Scheduler (ReduceLROnPlateau)**
- Reduce LR la jumătate (`factor=0.5`) dacă validation loss nu scade
- `patience=3`: Așteaptă 3 epoci înainte de reducere
- `verbose=True`: Afișează când se schimbă LR
- Ajută modelul să iasă din minime locale
- **Example output**: `Learning rate schimbat: 0.001000 -> 0.000500`

**Early Stopping**
- Oprește antrenarea automat când nu mai există îmbunătățiri
- `patience=10`: Numărul de epoci fără îmbunătățire înainte de oprire
- `min_delta=0.001`: Îmbunătățirea minimă considerată semnificativă
- Previne overfitting-ul și economisește timp
- **Example output**:
  ```
  Early stopping: 1/10 epoci fără îmbunătățire
  ...
  Early stopping activat la epoca 15
  Nu s-au observat îmbunătățiri timp de 10 epoci
  ```

### 2.6 Vizualizări XAI

#### Saliency Maps
- Gradient-based method
- Calculează gradientul imaginii față de clasa țintă
- Evidențiază ce regiuni influențează predicția

#### SHAP (SHapley Additive exPlanations)
- DeepExplainer pentru rețele neuronale
- Atribuie scor de importanță fiecărui pixel
- Roșu = contribuție pozitivă, Albastru = negativă

### 2.7 Configurare Extensibilă

```python
class Config:
    # Pentru 9 clase, schimbă:
    SELECTED_CLASSES = None  # În loc de lista cu 5 clase
    
    # Ajustare hiperparametri:
    BATCH_SIZE = 32          # Default: 16
    EPOCHS = 50              # Default: 30
    LEARNING_RATE = 0.001    # Default: 0.001
    
    # Early Stopping
    EARLY_STOPPING_PATIENCE = 10     # Default: 10 epoci
    EARLY_STOPPING_MIN_DELTA = 0.001  # Default: 0.001
```

### 2.8 Debugging Tips

**Problemă**: `IndexError: Target X is out of bounds`
- **Cauză**: Etichetele nu sunt remapate corect
- **Soluție**: Verifică `FilteredDataset` remap-ează la [0, num_classes-1]

**Problemă**: Out of memory
- **Soluție**: Reduce `BATCH_SIZE` la 8 sau 4

**Problemă**: Overfitting (validation loss crește, train scade)
- **Soluție**: 
  - Crește Dropout (0.5 → 0.7)
  - Adaugă mai multă augmentare
  - Reduce numărul de parametri
  - Early Stopping va opri automat antrenarea

**Problemă**: Early Stopping oprește prea devreme
- **Soluție**: Crește `EARLY_STOPPING_PATIENCE` (ex: 15 sau 20)

**Problemă**: Learning Rate scade prea repede
- **Soluție**: Crește `patience` în ReduceLROnPlateau (ex: 5)

---

## Resurse Utile

- **PyTorch Docs**: https://pytorch.org/docs/
- **TensorBoard**: https://www.tensorflow.org/tensorboard
- **MLflow**: https://mlflow.org/docs/latest/index.html
- **SHAP**: https://shap.readthedocs.io/
