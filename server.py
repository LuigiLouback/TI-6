from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import torch
from torchvision import transforms, models
import torch.nn as nn
from PIL import Image
import io
import os
import uvicorn

class TextureBranch(nn.Module):
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
      idx = var.argmin(dim=1)
      selected = torch.stack([patches[b,:,i] for b,i in enumerate(idx)])
      return self.features(selected).view(B, -1)

class ResNet50_TextureNet(nn.Module):
  def __init__(self, pretrained=False, num_classes=2):
      super().__init__()
      base = models.resnet50(weights=None)
      self.backbone = nn.Sequential(*list(base.children())[:-1])
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

print("Carregando modelo...")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ResNet50_TextureNet(pretrained=False, num_classes=2)

weights_path = "ai_vs_human_weights.pt"
if os.path.exists(weights_path):
  model.load_state_dict(torch.load(weights_path, map_location=device))
  model.to(device).eval()
  print(f" Modelo carregado! (Device: {device})")
else:
  print(f" Arquivo {weights_path} não encontrado!")
  exit(1)

transform = transforms.Compose([
  transforms.Resize((224, 224)),
  transforms.ToTensor(),
  transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

app = FastAPI(
  title="AI Art Classifier",
  description="Classifica se arte foi criada por humano ou IA",
  version="1.0.0"
)

app.add_middleware(
  CORSMiddleware,
  allow_origins=["*"],
  allow_credentials=True,
  allow_methods=["*"],
  allow_headers=["*"],
)

@app.get("/")
def root():
  return {
      "status": "online",
      "model": "ResNet50 + Texture Branch",
      "device": str(device),
      "endpoints": {
          "/classify": "POST - Upload de imagem",
          "/health": "GET - Status do servidor"
      }
  }

@app.get("/health")
def health():
  return {"status": "healthy", "device": str(device)}

@app.post("/classify")
async def classify_image(file: UploadFile = File(...)):
  """Classifica uma imagem como arte humana ou IA"""
  try:
      print("testei o classify")
      contents = await file.read()
      image = Image.open(io.BytesIO(contents)).convert('RGB')
      
      image_tensor = transform(image).unsqueeze(0).to(device)
      
      with torch.no_grad():
          logits = model(image_tensor)
          probs = torch.nn.functional.softmax(logits, dim=1)
          pred_class = logits.argmax(1).item()
      
      class_names = ["human", "ai"]
      class_labels = [" Arte Humana", " Arte gerada por IA"]
      confidence = probs[0][pred_class].item() * 100
      
      if confidence > 90:
          certainty = "muito alta"
      elif confidence > 75:
          certainty = "alta"
      elif confidence > 60:
          certainty = "moderada"
      else:
          certainty = "baixa"
      
      return {
          "success": True,
          "classification": class_names[pred_class],
          "label": class_labels[pred_class],
          "confidence_percentage": round(confidence, 2),
          "certainty_level": certainty,
          "probabilities": {
              "human": round(probs[0][0].item() * 100, 2),
              "ai": round(probs[0][1].item() * 100, 2)
          },
          "analysis": (
              f"Classificado como **{class_labels[pred_class]}** "
              f"com {certainty} confiança ({confidence:.1f}%). "
              f"Probabilidades: Humano={probs[0][0].item()*100:.1f}%, "
              f"IA={probs[0][1].item()*100:.1f}%."
          )
      }
      
  except Exception as e:
      raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
  port = int(os.environ.get("PORT", 8001))
  uvicorn.run(app, host="0.0.0.0", port=port)