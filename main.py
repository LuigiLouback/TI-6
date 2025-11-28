import os
import random
from matplotlib import pyplot as plt
import cv2
import numpy as np
import pandas as pd

import tensorflow as tf
from tensorflow import keras

from tensorflow.keras import layers, models, optimizers, callbacks
from tensorflow.keras.utils import image_dataset_from_directory
from tensorflow.keras.preprocessing.image import ImageDataGenerator

import keras_tuner as kt
import kagglehub

path = kagglehub.dataset_download("ravidussilva/real-ai-art")

print("Path to dataset files:", path)

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os
import glob
from concurrent.futures import ThreadPoolExecutor

class ArtDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.transform = transform
        self.samples = []

        def scan_folder(folder):
            folder_path = os.path.join(root_dir, folder)
            if not os.path.isdir(folder_path):
                return []
            
            # 0 = humano, 1 = IA
            label = 0 if not folder.startswith('AI_') else 1
            
            folder_samples = []
            for img_file in os.listdir(folder_path):
                if img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    img_path = os.path.join(folder_path, img_file)
                    folder_samples.append((img_path, label))
            return folder_samples

        # Escaneamento paralelo das pastas
        folders = [f for f in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, f))]
        with ThreadPoolExecutor(max_workers=4) as executor:
            results = list(executor.map(scan_folder, folders))
            self.samples = [item for sublist in results for item in sublist]

        print(f"📊 Total de imagens encontradas: {len(self.samples)}")
        human_count = sum(1 for _, label in self.samples if label == 0)
        ai_count = len(self.samples) - human_count
        print(f"👥 Humanas: {human_count}")
        print(f"🤖 IA: {ai_count}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image, label

# Normalização ImageNet
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

train_dir = os.path.join(path, "Real_AI_SD_LD_Dataset", "train")
test_dir  = os.path.join(path, "Real_AI_SD_LD_Dataset", "test")

print("🔄 Carregando dataset de TREINO...")
train_dataset = ArtDataset(train_dir, transform=transform)

print("\n🔄 Carregando dataset de TESTE...")
test_dataset = ArtDataset(test_dir, transform=transform)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_workers = 4

# DataLoader com otimizações de paralelismo
train_loader = DataLoader(
    train_dataset, 
    batch_size=32, 
    shuffle=True,
    num_workers=num_workers,
    pin_memory=(device.type == "cuda"),
    persistent_workers=(num_workers > 0),
    prefetch_factor=4 if num_workers > 0 else 2,
    multiprocessing_context='spawn' if num_workers > 0 else None
)

test_loader = DataLoader(
    test_dataset, 
    batch_size=32, 
    shuffle=False,
    num_workers=num_workers,
    pin_memory=(device.type == "cuda"),
    persistent_workers=(num_workers > 0),
    prefetch_factor=4 if num_workers > 0 else 2,
    multiprocessing_context='spawn' if num_workers > 0 else None
)

print("\n✅ PRONTO! Datasets criados.")
print(f"📈 Train: {len(train_dataset)} imagens")
print(f"📊 Test: {len(test_dataset)} imagens")

print("\n🧪 TESTE: Carregando primeira imagem...")
first_image, first_label = train_dataset[0]
print(f"Formato da imagem: {first_image.shape}")
print(f"Label: {first_label} ({'Humano' if first_label == 0 else 'IA'})")

import matplotlib.pyplot as plt
import numpy as np
import random

def plot_from_dataset(dataset, k=9, title="Imagens"):
    indices = random.choices(range(len(dataset)), k=k)

    cols = int(np.ceil(np.sqrt(k)))
    rows = int(np.ceil(k / cols))

    fig, axes = plt.subplots(rows, cols, figsize=(12, 8))
    if k == 1:
        axes = [axes]
    else:
        axes = axes.flatten() if hasattr(axes, 'flatten') else axes

    for i, idx in enumerate(indices):
        image_tensor, label = dataset[idx]

        img = image_tensor.permute(1, 2, 0).numpy()
        img = img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
        img = np.clip(img, 0, 1)

        axes[i].imshow(img)
        axes[i].axis('off')

        label_text = "👥 Humano" if label == 0 else "🤖 IA"
        axes[i].set_title(f"{label_text} #{idx}", fontsize=10)

    for i in range(k, len(axes)):
        axes[i].axis('off')

    plt.suptitle(title, fontsize=14)
    plt.tight_layout()
    plt.show()

print("🎨 Amostras aleatórias do dataset de TREINO:")
plot_from_dataset(train_dataset, k=9, title="Train Dataset - Humano vs IA")

print("\n🧪 Amostras aleatórias do dataset de TESTE:")
plot_from_dataset(test_dataset, k=6, title="Test Dataset - Humano vs IA")

import torch
from torch.utils.data import Subset
import numpy as np

def balance_dataset(dataset, target_count_per_class=5000):
    human_indices = [i for i, (_, label) in enumerate(dataset.samples) if label == 0]
    ai_indices = [i for i, (_, label) in enumerate(dataset.samples) if label == 1]

    print(f"📊 Antes do balanceamento:")
    print(f"👥 Humanas: {len(human_indices):,}")
    print(f"🤖 IA: {len(ai_indices):,}")

    np.random.seed(123)  # reprodutibilidade

    human_selected = np.random.choice(human_indices,
                                    min(target_count_per_class, len(human_indices)),
                                    replace=False)
    ai_selected = np.random.choice(ai_indices,
                                 min(target_count_per_class, len(ai_indices)),
                                 replace=False)

    balanced_indices = np.concatenate([human_selected, ai_selected])
    np.random.shuffle(balanced_indices)

    print(f"\n📊 Após balanceamento:")
    print(f"👥 Humanas: {len(human_selected):,}")
    print(f"🤖 IA: {len(ai_selected):,}")
    print(f"📈 Total: {len(balanced_indices):,}")

    return balanced_indices

balanced_indices = balance_dataset(train_dataset, target_count_per_class=50000)
balanced_train_dataset = Subset(train_dataset, balanced_indices)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_workers = 4

balanced_train_loader = DataLoader(
    balanced_train_dataset,
    batch_size=32,
    shuffle=True,
    num_workers=num_workers,
    pin_memory=(device.type == "cuda"),
    persistent_workers=(num_workers > 0),
    prefetch_factor=4 if num_workers > 0 else 2,
    multiprocessing_context='spawn' if num_workers > 0 else None
)

print(f"\n✅ Dataset balanceado criado!")
print(f"📊 Tamanho: {len(balanced_train_dataset):,} imagens")

human_count = sum(1 for i in balanced_indices
             if train_dataset.samples[i][1] == 0)
ai_count = len(balanced_indices) - human_count

print(f"👥 Humanas no dataset balanceado: {human_count:,}")
print(f"🤖 IA no dataset balanceado: {ai_count:,}")

import os

IMG_EXT = ('.jpg','.jpeg','.png','.bmp','.webp')

def list_images(root):
    out = []
    for d in os.listdir(root):
        dp = os.path.join(root, d)
        if os.path.isdir(dp):
            for f in os.listdir(dp):
                if f.lower().endswith(IMG_EXT):
                    out.append(os.path.join(dp, f))
    return out

def overlap_by_basename(train_dir, test_dir, top_k=20):
    tr = list_images(train_dir)
    te = list_images(test_dir)
    tr_base = {os.path.basename(p) for p in tr}
    te_base = {os.path.basename(p) for p in te}
    inter = tr_base & te_base
    print(f"⚠️  Basenames em comum entre train/test: {len(inter)}")
    if inter:
        print("Exemplos:", list(sorted(inter))[:top_k])

overlap_by_basename(train_dir, test_dir)

import matplotlib.pyplot as plt
import numpy as np
import random

def plot_from_dataset(dataset, k=9, title="Imagens"):
    indices = random.choices(range(len(dataset)), k=k)

    cols = int(np.ceil(np.sqrt(k)))
    rows = int(np.ceil(k / cols))

    fig, axes = plt.subplots(rows, cols, figsize=(12, 8))
    if k == 1:
        axes = [axes]
    else:
        axes = axes.flatten() if hasattr(axes, 'flatten') else axes

    for i, idx in enumerate(indices):
        image_tensor, label = dataset[idx]

        img = image_tensor.permute(1, 2, 0).numpy()
        img = img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
        img = np.clip(img, 0, 1)

        axes[i].imshow(img)
        axes[i].axis('off')

        label_text = "👥 Humano" if label == 0 else "🤖 IA"
        axes[i].set_title(f"{label_text} #{idx}", fontsize=10)

    for i in range(k, len(axes)):
        axes[i].axis('off')

    plt.suptitle(title, fontsize=14)
    plt.tight_layout()
    plt.show()

def plot_human_only(dataset, k=9):
    human_indices = [i for i, (_, label) in enumerate(dataset.samples) if label == 0]
    selected_indices = random.choices(human_indices, k=k)

    cols = int(np.ceil(np.sqrt(k)))
    rows = int(np.ceil(k / cols))

    fig, axes = plt.subplots(rows, cols, figsize=(12, 8))
    if k == 1:
        axes = [axes]
    else:
        axes = axes.flatten()

    for i, idx in enumerate(selected_indices):
        image_tensor, _ = dataset[idx]

        img = image_tensor.permute(1, 2, 0).numpy()
        img = img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
        img = np.clip(img, 0, 1)

        axes[i].imshow(img)
        axes[i].axis('off')
        axes[i].set_title(f"👥 Arte Humana #{idx}", fontsize=10)

    for i in range(k, len(axes)):
        axes[i].axis('off')

    plt.suptitle("🎨 Arte Criada por Humanos", fontsize=14)
    plt.tight_layout()
    plt.show()

def plot_ai_only(dataset, k=9):
    ai_indices = [i for i, (_, label) in enumerate(dataset.samples) if label == 1]
    selected_indices = random.choices(ai_indices, k=k)

    cols = int(np.ceil(np.sqrt(k)))
    rows = int(np.ceil(k / cols))

    fig, axes = plt.subplots(rows, cols, figsize=(12, 8))
    if k == 1:
        axes = [axes]
    else:
        axes = axes.flatten()

    for i, idx in enumerate(selected_indices):
        image_tensor, _ = dataset[idx]

        img = image_tensor.permute(1, 2, 0).numpy()
        img = img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
        img = np.clip(img, 0, 1)

        axes[i].imshow(img)
        axes[i].axis('off')
        axes[i].set_title(f"🤖 Arte IA #{idx}", fontsize=10)

    for i in range(k, len(axes)):
        axes[i].axis('off')

    plt.suptitle("🤖 Arte Gerada por IA", fontsize=14)
    plt.tight_layout()
    plt.show()

def plot_comparison(dataset, k_per_class=6):
    human_indices = [i for i, (_, label) in enumerate(dataset.samples) if label == 0]
    ai_indices = [i for i, (_, label) in enumerate(dataset.samples) if label == 1]

    human_selected = random.choices(human_indices, k=k_per_class)
    ai_selected = random.choices(ai_indices, k=k_per_class)

    fig, axes = plt.subplots(2, k_per_class, figsize=(15, 6))

    for i, idx in enumerate(human_selected):
        image_tensor, _ = dataset[idx]
        img = image_tensor.permute(1, 2, 0).numpy()
        img = img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
        img = np.clip(img, 0, 1)

        axes[0, i].imshow(img)
        axes[0, i].axis('off')
        axes[0, i].set_title(f"👥 Humano", fontsize=10)

    for i, idx in enumerate(ai_selected):
        image_tensor, _ = dataset[idx]
        img = image_tensor.permute(1, 2, 0).numpy()
        img = img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
        img = np.clip(img, 0, 1)

        axes[1, i].imshow(img)
        axes[1, i].axis('off')
        axes[1, i].set_title(f"🤖 IA", fontsize=10)

    plt.suptitle("🆚 Comparação: Arte Humana vs IA", fontsize=16)
    plt.tight_layout()
    plt.show()

print("🎨 1. Amostras aleatórias (misturadas):")
plot_from_dataset(train_dataset, k=9, title="Train Dataset - Humano vs IA")

print("\n👥 2. Apenas arte HUMANA:")
plot_human_only(train_dataset, k=9)

print("\n🤖 3. Apenas arte IA:")
plot_ai_only(train_dataset, k=9)

print("\n🆚 4. Comparação lado a lado:")
plot_comparison(train_dataset, k_per_class=6)

print("\n🧪 5. Amostras do dataset de TESTE:")
plot_from_dataset(test_dataset, k=6, title="Test Dataset")

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

class TextureBranch(nn.Module):
    """Extrai features de textura local selecionando patch com menor variância"""
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1)
        )
        self.out_dim = 128

    def forward(self, x):
        B, C, H, W = x.shape
        patch_size = H // 4
        stride = patch_size // 2
        patches = x.unfold(2, patch_size, stride).unfold(3, patch_size, stride)
        patches = patches.contiguous().view(B, C, -1, patch_size, patch_size)
        var = patches.var(dim=(1,3,4))
        idx = var.argmin(dim=1)  # patch mais homogêneo
        selected = torch.stack([patches[b,:,i] for b,i in enumerate(idx)])

        return self.features(selected).view(B, -1)


class ResNet50_TextureNet(nn.Module):
    """ResNet50 com branch adicional de textura para melhorar detecção de arte IA"""
    def __init__(self, pretrained=True, num_classes=2):
        super().__init__()
        base = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None)
        self.backbone = nn.Sequential(*list(base.children())[:-1])  # remove fc
        self.global_out = base.fc.in_features

        self.texture_branch = TextureBranch()

        combined_dim = self.global_out + self.texture_branch.out_dim
        self.classifier = nn.Sequential(
            nn.Linear(combined_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        global_feat = self.backbone(x).flatten(1)
        texture_feat = self.texture_branch(x)
        feats = torch.cat([global_feat, texture_feat], dim=1)
        return self.classifier(feats)

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from tqdm.auto import tqdm
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_score, recall_score, f1_score
import numpy as np
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

model = ResNet50_TextureNet(pretrained=True, num_classes=2).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

EPOCHS = 1

for epoch in range(EPOCHS):
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    for images, labels in tqdm(balanced_train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        preds = outputs.argmax(1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    print(f"Epoch {epoch+1}: loss={running_loss/total:.4f} acc={correct/total:.4f}")

torch.save(model.state_dict(), "ai_vs_human_weights.pt")

model.eval()
all_preds, all_labels = [], []

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        preds = model(images).argmax(1)
        all_preds.append(preds.cpu().numpy())
        all_labels.append(labels.cpu().numpy())

y_pred = np.concatenate(all_preds)
y_true = np.concatenate(all_labels)
print("\nClassification Report:")
print(classification_report(y_true, y_pred, target_names=["Human","AI"], zero_division=0))

print("Accuracy:", accuracy_score(y_true, y_pred))
print("Precision:", precision_score(y_true, y_pred, average="binary"))
print("Recall:", recall_score(y_true, y_pred, average="binary"))
print("F1:", f1_score(y_true, y_pred, average="binary"))

cm = confusion_matrix(y_true, y_pred)
print("\nConfusion Matrix:\n", cm)

# Plot da matriz de confusão
fig, ax = plt.subplots()
im = ax.imshow(cm, cmap="Blues")
ax.set_xticks([0,1]); ax.set_yticks([0,1])
ax.set_xticklabels(["Human","AI"]); ax.set_yticklabels(["Human","AI"])
ax.set_xlabel("Predito"); ax.set_ylabel("Verdadeiro")
ax.set_title("Matriz de Confusão — Human vs AI")
for i in range(2):
    for j in range(2):
        ax.text(j,i,cm[i,j],ha="center",va="center",color="red")
plt.show()

torch.save(model.state_dict(), "ai_vs_human_weights.pt")

import kagglehub

path = kagglehub.dataset_download("kausthubkannan/ai-and-human-art-classification")

print("Path to dataset files:", path)

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Usando train porque tem labels
test2_dir = os.path.join(path, "ai_art_classification", "train")

print("\n🔄 Carregando dataset de TESTE...")
test2_dataset = ArtDataset(test2_dir, transform=transform)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_workers = 4

test2_loader = DataLoader(
    test2_dataset, 
    batch_size=32, 
    shuffle=False,
    num_workers=num_workers,
    pin_memory=(device.type == "cuda"),
    persistent_workers=(num_workers > 0),
    prefetch_factor=4 if num_workers > 0 else 2,
    multiprocessing_context='spawn' if num_workers > 0 else None
)

print("\n✅ PRONTO! Datasets criados.")
print(f"📊 Test: {len(test2_dataset)} imagens")

print("\n🧪 TESTE: Carregando primeira imagem...")
first_image, first_label = test2_dataset[0]
print(f"Formato da imagem: {first_image.shape}")
print(f"Label: {first_label} ({'Humano' if first_label == 0 else 'IA'})")

import torch
import torch.nn as nn
from torchvision import models

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = ResNet50_TextureNet(pretrained=False, num_classes=2)
state = torch.load("ai_vs_human_weights.pt", map_location=device)

# Remove prefixo "module." se modelo foi salvo com DataParallel
if any(k.startswith("module.") for k in state.keys()):
    state = {k.replace("module.", "", 1): v for k, v in state.items()}

model.load_state_dict(state)
model.to(device).eval()

missing, unexpected = model.load_state_dict(state, strict=False)
print("missing:", missing, "| unexpected:", unexpected)

model.to(device).eval()
print("✅ Modelo carregado e pronto para inferência!")

import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

y_true, y_pred = [], []

model.eval()
with torch.no_grad():
    for images, labels in test2_loader:
        images = images.to(device)
        logits = model(images)
        preds = logits.argmax(dim=1).cpu().numpy()
        y_pred.append(preds)
        y_true.append(labels.numpy())

y_pred = np.concatenate(y_pred)
y_true = np.concatenate(y_true)
print("\nClassification Report:")
print(classification_report(y_true, y_pred, target_names=["Human","AI"], zero_division=0))

print("Accuracy:",  accuracy_score(y_true, y_pred))
print("Precision:", precision_score(y_true, y_pred, average="binary", pos_label=1))
print("Recall:",    recall_score(y_true, y_pred, average="binary", pos_label=1))
print("F1:",        f1_score(y_true, y_pred, average="binary", pos_label=1))

cm = confusion_matrix(y_true, y_pred)
print("\nConfusion Matrix:\n", cm)

import matplotlib.pyplot as plt
fig, ax = plt.subplots()
im = ax.imshow(cm, cmap="Blues")
ax.set_xticks([0,1]); ax.set_yticks([0,1])
ax.set_xticklabels(["Human","AI"]); ax.set_yticklabels(["Human","AI"])
ax.set_xlabel("Predito"); ax.set_ylabel("Verdadeiro")
ax.set_title("Matriz de Confusão — Human vs AI")
for i in range(2):
    for j in range(2):
        ax.text(j, i, cm[i, j], ha="center", va="center", color="red")
plt.colorbar(im, ax=ax)
plt.show()

