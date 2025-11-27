"""
Benchmarks de Escalabilidade Forte e Fraca para Treinamento
Testa diferentes configurações de num_workers no DataLoader
"""

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path
from statistics import mean
from typing import Dict, Iterable, List, Sequence, Optional
from itertools import islice
from itertools import cycle as infinite_cycle

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import models, transforms
from PIL import Image
import numpy as np


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
        stride = max(patch_size // 2, 1)
        patches = x.unfold(2, patch_size, stride).unfold(3, patch_size, stride)
        patches = patches.contiguous().view(B, C, -1, patch_size, patch_size)
        var = patches.var(dim=(1, 3, 4))
        idx = var.argmin(dim=1)
        selected = torch.stack([patches[b, :, i] for b, i in enumerate(idx)])
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


class ArtDataset(Dataset):
    def __init__(self, root_dir, transform=None, max_samples_per_class=None):
        """
        Args:
            root_dir: Diretório com as imagens
            transform: Transformações a aplicar
            max_samples_per_class: Limite de imagens por classe (None = todas)
        """
        self.transform = transform
        self.samples = []
        
        human_samples = []
        ai_samples = []

        for folder in os.listdir(root_dir):
            folder_path = os.path.join(root_dir, folder)
            if os.path.isdir(folder_path):
                label = 0 if not folder.startswith('AI_') else 1
                for img_file in os.listdir(folder_path):
                    if img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                        img_path = os.path.join(folder_path, img_file)
                        if label == 0:
                            human_samples.append((img_path, label))
                        else:
                            ai_samples.append((img_path, label))
        
        if max_samples_per_class is not None:
            random.seed(42)
            random.shuffle(human_samples)
            random.shuffle(ai_samples)
            human_samples = human_samples[:max_samples_per_class]
            ai_samples = ai_samples[:max_samples_per_class]
        
        self.samples = human_samples + ai_samples
        random.shuffle(self.samples)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, label


def cycle_indices(indices: Sequence[int], total: int) -> List[int]:
    """Repete índices até atingir o tamanho desejado"""
    if total <= len(indices):
        return list(indices[:total])
    iterator = infinite_cycle(indices)
    return list(islice(iterator, total))


def run_training_once(
    train_loader: DataLoader,
    device: torch.device,
    epochs: int = 1
) -> float:
    """Executa uma época completa de treinamento e retorna o tempo gasto."""
    model = ResNet50_TextureNet(pretrained=False, num_classes=2).to(device)
    
    stream_compute = None
    stream_transfer = None
    if device.type == "cuda":
        stream_compute = torch.cuda.Stream()
        stream_transfer = torch.cuda.Stream()
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    start = time.perf_counter()
    
    for epoch in range(epochs):
        model.train()
        
        for images, labels in train_loader:
            if device.type == "cuda" and stream_transfer is not None and stream_compute is not None:
                with torch.cuda.stream(stream_transfer):
                    images = images.to(device, non_blocking=True)
                    labels = labels.to(device, non_blocking=True)
                
                torch.cuda.current_stream().wait_stream(stream_transfer)
                with torch.cuda.stream(stream_compute):
                    optimizer.zero_grad()
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()
                
                torch.cuda.current_stream().wait_stream(stream_compute)
            else:
                images = images.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
        
        if device.type == "cuda":
            torch.cuda.synchronize()
    
    elapsed = time.perf_counter() - start
    return elapsed


def run_training(
    dataset: Dataset,
    num_workers: int,
    batch_size: int,
    epochs: int,
    device: torch.device,
    repetitions: int,
    shuffle: bool = True
) -> Dict[str, float]:
    """Executa treinamento múltiplas vezes variando num_workers."""
    durations = []
    
    for rep in range(repetitions):
        print(f"  Repetição {rep + 1}/{repetitions}...", end=" ", flush=True)
        
        train_loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=(device.type == "cuda")
        )
        
        duration = run_training_once(train_loader, device, epochs)
        durations.append(duration)
        print(f"{duration:.2f}s")
    
    return {
        "time": mean(durations),
        "min_time": min(durations),
        "max_time": max(durations),
    }


def strong_scaling(
    dataset: Dataset,
    workers_list: Iterable[int],
    batch_size: int,
    epochs: int,
    device: torch.device,
    repetitions: int
) -> List[Dict[str, float]]:
    """Escalabilidade forte: mesmo dataset, diferentes num_workers."""
    print("\n" + "="*60)
    print("ESCALABILIDADE FORTE")
    print("="*60)
    print(f"Dataset fixo: {len(dataset)} imagens")
    print(f"Testando com diferentes números de workers...")
    print(f"Cada teste executa {epochs} época(s) completa(s)\n")
    
    first_worker = workers_list[0] if workers_list[0] > 0 else 1
    print(f"\n📊 Baseline (1 worker):")
    baseline = run_training(
        dataset, first_worker, batch_size, epochs, device, repetitions
    )["time"]
    
    results = []
    for workers in sorted(set(workers_list) | {0}):
        if workers == 0:
            workers = 1
        
        print(f"\n📊 Workers: {workers}")
        metrics = run_training(
            dataset, workers, batch_size, epochs, device, repetitions
        )
        
        speedup = baseline / metrics["time"]
        efficiency = speedup / workers
        
        results.append({
            "workers": workers,
            "time": metrics["time"],
            "min_time": metrics["min_time"],
            "max_time": metrics["max_time"],
            "speedup": speedup,
            "efficiency": efficiency,
        })
    
    return results


def weak_scaling(
    dataset: Dataset,
    workers_list: Iterable[int],
    batch_size: int,
    epochs: int,
    device: torch.device,
    repetitions: int
) -> List[Dict[str, float]]:
    """Escalabilidade fraca: dataset cresce proporcionalmente aos workers."""
    print("\n" + "="*60)
    print("ESCALABILIDADE FRACA")
    print("="*60)
    base_size = len(dataset)
    print(f"Dataset base: {base_size} imagens")
    print(f"Escalando dataset proporcionalmente aos workers...")
    print(f"Cada teste executa {epochs} época(s) completa(s)\n")
    
    all_indices = list(range(len(dataset)))
    
    baseline_dataset = Subset(dataset, all_indices[:base_size])
    print(f"\n📊 Baseline (1 worker, {base_size} imagens):")
    baseline = run_training(
        baseline_dataset, 1, batch_size, epochs, device, repetitions
    )["time"]
    
    results = []
    for workers in sorted(set(workers_list) | {1}):
        if workers == 0:
            workers = 1
        
        target_size = base_size * workers
        scaled_indices = cycle_indices(all_indices, target_size)
        scaled_dataset = Subset(dataset, scaled_indices)
        
        print(f"\n📊 Workers: {workers} | Dataset: {target_size} imagens ({workers}x)")
        metrics = run_training(
            scaled_dataset, workers, batch_size, epochs, device, repetitions
        )
        
        speedup = baseline / metrics["time"]
        efficiency = speedup / workers
        
        results.append({
            "workers": workers,
            "workload": target_size,
            "time": metrics["time"],
            "min_time": metrics["min_time"],
            "max_time": metrics["max_time"],
            "speedup": speedup,
            "efficiency": efficiency,
        })
    
    return results


def print_table(
    results: List[Dict[str, float]],
    title: str,
    include_workload: bool
) -> None:
    """Imprime tabela formatada com os resultados"""
    if not results:
        return
    
    headers = ["Workers", "Tempo (s)", "Speedup", "Eficiência", "Tempo Mín", "Tempo Máx"]
    if include_workload:
        headers.insert(1, "Carga")
    
    widths = [len(header) for header in headers]
    rows = []
    
    for item in sorted(results, key=lambda x: x["workers"]):
        row = [
            f"{item['workers']:>7}",
            f"{item['time']:.4f}",
            f"{item['speedup']:.2f}",
            f"{item['efficiency']:.2f}",
            f"{item['min_time']:.4f}",
            f"{item['max_time']:.4f}",
        ]
        if include_workload:
            row.insert(1, f"{int(item['workload']):>5}")
        rows.append(row)
        widths = [max(widths[i], len(row[i])) for i in range(len(headers))]
    
    separator = " | "
    print(f"\n{title}")
    print("-" * (sum(widths) + len(separator) * (len(headers) - 1)))
    header_row = separator.join(header.ljust(widths[i]) for i, header in enumerate(headers))
    print(header_row)
    print("-" * (sum(widths) + len(separator) * (len(headers) - 1)))
    for row in rows:
        print(separator.join(row[i].ljust(widths[i]) for i in range(len(headers))))
    print("-" * (sum(widths) + len(separator) * (len(headers) - 1)))


def parse_workers(spec: str) -> List[int]:
    """Parse string de workers (ex: '1,2,4,8')"""
    workers = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        workers.append(int(part))
    if not workers:
        raise ValueError("Lista de workers não pode ser vazia.")
    return workers


def save_report(
    path: Path,
    dataset_dir: Path,
    strong_results: List[Dict[str, float]],
    weak_results: List[Dict[str, float]],
    batch_size: int,
    epochs: int,
    device: str
) -> None:
    """Salva relatório em JSON"""
    report = {
        "dataset": str(dataset_dir.resolve()),
        "device": device,
        "batch_size": batch_size,
        "epochs": epochs,
        "model": "ResNet50 + Texture Branch",
        "strong_scaling": strong_results,
        "weak_scaling": weak_results,
    }
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")


def main(argv: Sequence[str]) -> int:
    parser = argparse.ArgumentParser(
        description="Benchmarks de escalabilidade forte e fraca para treinamento do classificador AI Art."
    )
    parser.add_argument(
        "--dataset",
        required=False,
        type=Path,
        default=None,
        help="Diretório com as imagens de treino. Se não fornecido, usa kagglehub para baixar automaticamente.",
    )
    parser.add_argument(
        "--workers",
        default="0,1,2,4",
        help="Lista de quantidades de workers separados por vírgula. Ex: 0,1,2,4,8 (0 = sequencial)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Tamanho do batch para treinamento.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=1,
        help="Número de épocas para cada teste.",
    )
    parser.add_argument(
        "--repetitions",
        type=int,
        default=1,
        help="Número de repetições para cada medição.",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=None,
        help="Caminho opcional para salvar o relatório em JSON.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Limite de imagens por classe (None = todas). Ex: 5000 para usar 5000 de cada classe.",
    )
    
    args = parser.parse_args(argv)
    
    workers_list = parse_workers(args.workers)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Se dataset não foi fornecido, usar kagglehub
    if args.dataset is None:
        print("\n📥 Dataset não fornecido. Baixando via kagglehub...")
        try:
            import kagglehub
            path = kagglehub.dataset_download("ravidussilva/real-ai-art")
            dataset_path = Path(path) / "Real_AI_SD_LD_Dataset" / "train"
            print(f"✅ Dataset baixado em: {dataset_path}")
        except ImportError:
            print("❌ Erro: kagglehub não está instalado!")
            print("   Instale com: pip install kagglehub")
            return 1
        except Exception as e:
            print(f"❌ Erro ao baixar dataset: {e}")
            return 1
    else:
        dataset_path = args.dataset
    
    print("\n" + "="*60)
    print("BENCHMARK DE COMPUTAÇÃO PARALELA - TREINAMENTO")
    print("="*60)
    print(f"Dataset: {dataset_path.resolve()}")
    print(f"Device: {device}")
    print(f"Batch Size: {args.batch_size}")
    print(f"Épocas: {args.epochs}")
    print(f"Repetições: {args.repetitions}")
    print(f"Workers testados: {workers_list}")
    print("="*60)
    
    # Carregar dataset
    print("\n📂 Carregando dataset...")
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    train_dataset = ArtDataset(dataset_path, transform=transform, max_samples_per_class=args.max_samples)
    total_images = len(train_dataset)
    if args.max_samples:
        print(f"✅ Dataset carregado: {total_images} imagens ({args.max_samples} de cada classe)")
    else:
        print(f"✅ Dataset carregado: {total_images} imagens (todas)")
    
    # Executar benchmarks
    strong_results = strong_scaling(
        train_dataset, workers_list, args.batch_size,
        args.epochs, device, args.repetitions
    )
    
    weak_results = weak_scaling(
        train_dataset, workers_list, args.batch_size,
        args.epochs, device, args.repetitions
    )
    
    # Imprimir resultados
    print_table(strong_results, "Escalabilidade Forte", include_workload=False)
    print_table(weak_results, "Escalabilidade Fraca", include_workload=True)
    
    # Salvar relatório
    if args.report:
        save_report(
            args.report, dataset_path, strong_results, weak_results,
            args.batch_size, args.epochs, str(device)
        )
        print(f"\n✅ Relatório salvo em: {args.report.resolve()}")
    
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

