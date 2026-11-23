"""Détecteur minimal : feature maps -> prédiction par cellule -> NMS -> une boîte (+ la forme).

Image 64x64 -> 3 blocs conv stride 2 -> feature map 32x8x8 (stride 8).
Une conv 1x1 prédit pour chaque cellule 5 valeurs : (objectness, dx, dy, w, h)
  - dx, dy in [0,1] : décalage du centre dans la cellule
  - w, h in [0,1]   : taille relative à l'image

OPTION B (plusieurs classes de formes) : la tête prédit en plus 3 scores de classe
(cercle, carré, triangle), soit 8 valeurs par cellule. Les lignes modifiées sont
marquées « OPTION B ».
"""

from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.utils.data import Dataset

STRIDE = 8
CLASSES = ("cercle", "carré", "triangle")   # OPTION B : mêmes numéros que dataset_generator_2 (0, 1, 2)
NUM_CLASSES = len(CLASSES)


def iou(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """IoU entre une boîte a (4,) et des boîtes b (N, 4)."""
    inter_w = (torch.minimum(a[2], b[:, 2]) - torch.maximum(a[0], b[:, 0])).clamp(min=0)
    inter_h = (torch.minimum(a[3], b[:, 3]) - torch.maximum(a[1], b[:, 1])).clamp(min=0)
    inter = inter_w * inter_h
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / (area_a + area_b - inter).clamp(min=1e-8)


def nms(boxes: torch.Tensor, scores: torch.Tensor, iou_threshold: float = 0.5) -> list[int]:
    """Garde la meilleure boîte, supprime celles qui la chevauchent, recommence et retourne les indices conservés."""
    order = scores.argsort(descending=True)
    kept = []
    while len(order):
        best = order[0]
        kept.append(int(best))
        order = order[1:]
        order = order[iou(boxes[best], boxes[order]) < iou_threshold]
    return kept


def code(cx: float, cy: float, w: float, h: float, classe: int | None = None, size: int = 64) -> torch.Tensor:
    """Boîte [cx, cy, w, h] en pixels -> cible (5, G, G) pour l'entraînement.

    OPTION B : si on donne la classe, la cible a un 6e canal qui contient le numéro
    de la classe dans la cellule du centre -> (6, G, G).
    """
    grid = size // STRIDE
    gx, gy = cx / size * grid, cy / size * grid
    col, row = min(int(gx), grid - 1), min(int(gy), grid - 1)
    values = [1.0, gx - col, gy - row, w / size, h / size]
    if classe is not None:                      # OPTION B
        values.append(float(classe))
    target = torch.zeros(len(values), grid, grid)
    target[:, row, col] = torch.tensor(values)
    return target


def decode(logits: torch.Tensor, size: int = 64) -> tuple[torch.Tensor, torch.Tensor]:
    """(5+, G, G) -> boîtes (G*G, 4) en pixels [x1, y1, x2, y2] et scores (G*G,).

    Inchangée pour l'option B : elle n'utilise que les canaux 0 à 4 (la boîte).
    """
    grid = logits.shape[-1]
    p = torch.sigmoid(logits[:5])
    rows, cols = torch.meshgrid(torch.arange(grid), torch.arange(grid), indexing="ij")
    cx = (cols + p[1]) * STRIDE
    cy = (rows + p[2]) * STRIDE
    w, h = p[3] * size, p[4] * size
    boxes = torch.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], dim=-1)
    return boxes.reshape(-1, 4), p[0].reshape(-1)


class ShapeDataset(Dataset):
    """Cible (5 ou 6, G, G) : tout à 0 sauf la cellule contenant le centre."""

    def __init__(self, folder: Path, size: int = 64) -> None:
        self.paths = sorted(Path(folder).glob("*.png"))
        if not self.paths:
            raise FileNotFoundError(f"Aucune image dans {folder} (lancer dataset_generator).")
        self.grid = size // STRIDE

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, i: int) -> tuple[torch.Tensor, torch.Tensor]:
        path = self.paths[i]
        image = torch.from_numpy(
            np.asarray(Image.open(path).convert("RGB"), dtype=np.float32).copy()
        ).permute(2, 0, 1) / 255.0
        values = path.with_suffix(".txt").read_text().split()
        cx, cy, w, h = map(float, values[:4])
        classe = int(values[4]) if len(values) == 5 else None    # OPTION B : 5e valeur = classe
        target = code(cx, cy, w, h, classe, size=image.shape[-1])
        return image, target


class Detector(nn.Module):
    def __init__(self, num_classes: int = 0) -> None:
        """num_classes = 0 : détecteur du TP (5 sorties). num_classes = 3 : option B (8 sorties)."""
        super().__init__()
        self.num_classes = num_classes
        # Backbone : 3 blocs (conv 3x3 stride 2 + BatchNorm + ReLU) : 64 -> 32 -> 16 -> 8.
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=3, stride=2, padding=1),    # (B, 8, 32, 32)
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.Conv2d(8, 16, kernel_size=3, stride=2, padding=1),   # (B, 16, 16, 16)
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),  # (B, 32, 8, 8)
            nn.BatchNorm2d(32),
            nn.ReLU(),
        )
        # Tête : conv 1x1 = la même couche dense appliquée à chaque cellule.
        # OPTION B : 5 + 3 = 8 sorties (objectness, dx, dy, w, h, cercle, carré, triangle).
        self.head = nn.Conv2d(32, 5 + num_classes, kernel_size=1)

    def feature_maps(self, images: torch.Tensor) -> torch.Tensor:
        """(B, 3, 64, 64) -> (B, 32, 8, 8)."""
        return self.backbone(images)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """(B, 3, 64, 64) -> logits (B, 5 + num_classes, 8, 8)."""
        return self.head(self.feature_maps(images))


def compute_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """BCE sur l'objectness de toutes les cellules + MSE sur la boîte de la cellule positive.

    OPTION B : + entropie croisée sur les 3 scores de classe de la cellule positive.
    """
    obj_loss = nn.functional.binary_cross_entropy_with_logits(
        logits[:, 0], target[:, 0], pos_weight=torch.tensor(10.0))
    positive = target[:, 0] > 0.5  # (B, G, G)
    pred_box = torch.sigmoid(logits[:, 1:5]).permute(0, 2, 3, 1)[positive]
    true_box = target[:, 1:5].permute(0, 2, 3, 1)[positive]
    loss = obj_loss + 10.0 * nn.functional.mse_loss(pred_box, true_box)

    if logits.shape[1] > 5:                                                  # OPTION B
        class_logits = logits[:, 5:].permute(0, 2, 3, 1)[positive]          # (B, 3)
        true_class = target[:, 5][positive].long()                          # (B,)
        loss = loss + nn.functional.cross_entropy(class_logits, true_class)
    return loss


def predict(logits: torch.Tensor, size: int = 64) -> tuple[torch.Tensor, float, int | None]:
    """Une seule figure par image : la boîte de plus fort score après NMS.

    OPTION B : la classe est l'argmax des 3 scores de classe de cette cellule (None pour le TP).
    """
    boxes, scores = decode(logits, size)
    best = nms(boxes, scores)[0]
    classe = None
    if logits.shape[0] > 5:                                                  # OPTION B
        class_scores = logits[5:].reshape(logits.shape[0] - 5, -1)          # (3, G*G)
        classe = int(class_scores[:, best].argmax())
    return boxes[best], float(scores[best]), classe
