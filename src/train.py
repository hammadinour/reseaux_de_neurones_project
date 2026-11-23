"""Entraîne le détecteur, évalue (IoU moyen, % IoU > 0.5, % bonne classe)."""

import random

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from src.dataset_generator_2 import make_example
from src.detector import NUM_CLASSES, Detector, code, compute_loss, iou, predict


def make_dataset(count: int, seed: int, size: int = 64) -> TensorDataset:
    """Génère `count` images (une forme et une couleur au hasard) et leurs cibles (6, G, G)."""
    rng = random.Random(seed)
    images, targets = [], []
    for _ in range(count):
        image, label = make_example(size, rng)                  # label = "cx cy w h classe"
        cx, cy, w, h, classe = label.split()
        images.append(torch.from_numpy(np.asarray(image, dtype=np.float32).copy()).permute(2, 0, 1) / 255.0)
        targets.append(code(float(cx), float(cy), float(w), float(h), int(classe), size=size))
    return TensorDataset(torch.stack(images), torch.stack(targets))


def train(model: Detector, loader: DataLoader, optimizer: torch.optim.Optimizer) -> float:
    """Entraîne le modèle sur une époque et retourne la perte moyenne."""
    model.train()  # mode entraînement (BatchNorm utilise les statistiques du batch)
    total_loss = 0.0
    for images, targets in loader:
        optimizer.zero_grad()                     # 1. on remet les gradients à zéro
        logits = model(images)                    # 2. passe avant (forward)
        loss = compute_loss(logits, targets)      # 3. calcul de la perte
        loss.backward()                           # 4. rétropropagation
        optimizer.step()                          # 5. mise à jour des poids
        total_loss += loss.item() * images.size(0)
    return total_loss / len(loader.dataset)


def evaluate(model: Detector, loader: DataLoader, size: int = 64) -> dict:
    """Retourne l'IoU moyen, la proportion d'IoU > 0,5 et (option B) la proportion de bonne classe.

    Remplit aussi une matrice de confusion (lignes = vraie forme, colonnes = forme prédite).
    """
    model.eval()
    ious, good = [], []
    confusion = torch.zeros(NUM_CLASSES, NUM_CLASSES, dtype=torch.long)
    with torch.no_grad():
        for images, targets in loader:
            logits = model(images)
            grid = targets.shape[-1]
            for logit, target in zip(logits, targets):
                box, _, classe = predict(logit, size)
                row, col = divmod(int(target[0].argmax()), grid)
                _, dx, dy, w, h = target[:5, row, col].tolist()
                cx, cy = (col + dx) * size / grid, (row + dy) * size / grid
                truth = torch.tensor([cx - w * size / 2, cy - h * size / 2,
                                      cx + w * size / 2, cy + h * size / 2])
                ious.append(float(iou(box, truth[None])[0]))
                if classe is not None:                                       # OPTION B
                    true_class = int(target[5, row, col])
                    good.append(classe == true_class)
                    confusion[true_class, classe] += 1
    t = torch.tensor(ious)
    return {
        "iou": t.mean().item(),
        "iou50": (t > 0.5).float().mean().item(),
        "classe": torch.tensor(good, dtype=torch.float32).mean().item() if good else None,
        "confusion": confusion,
    }
