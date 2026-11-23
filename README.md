# Mini-projet option B : plusieurs classes de formes

## Lancer

```bash
pip install -r requirements.txt
jupyter notebook notebooks/mini_projet_optionB.ipynb     # ou ouvrir le notebook dans VS Code
```

Exécuter toutes les cellules dans l'ordre (≈ 5 à 10 min sur un PC portable).
Les figures sont aussi sauvegardées dans `outputs/` et les courbes TensorBoard dans `runs/`.

## Ce qui change par rapport au TP (lignes marquées « OPTION B » dans le code)

| Fichier | Fonction | Modification |
|---|---|---|
| `detector.py` | `code` | argument `classe` en plus : la cible a 6 canaux `[1, dx, dy, w̃, h̃, classe]` |
| `detector.py` | `ShapeDataset.__getitem__` | lit la 5e valeur du `.txt` (la classe) |
| `detector.py` | `Detector` | `num_classes` : tête `Conv2d(32, 5 + 3, 1)` = 8 sorties |
| `detector.py` | `compute_loss` | `+ CrossEntropy` sur `logits[:, 5:8]`, seulement dans la cellule positive |
| `detector.py` | `predict` | la forme = `argmax` des 3 scores de la meilleure cellule |
| `detector.py` | `nms` | seuil 0,5 (valeur du cours) au lieu de 1,1 |
| `train.py` | `make_dataset` | génère les données avec `dataset_generator_2` |
| `train.py` | `evaluate` | renvoie aussi le % de bonne classe et la matrice de confusion |
| `visualize.py` | `show_detections` | affiche le nom de la forme (en rouge si elle est fausse) |

`decode` et le backbone ne changent pas.

Perte : `L = BCE(objectness) + 10 · MSE(dx, dy, w, h) + CE(classe)`.
