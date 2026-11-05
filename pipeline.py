"""
pipeline.py — Código corregido para el estudio de valencia afectiva en imágenes publicitarias.

Reemplaza la lógica de Transfer_Learning_.ipynb y Lime_.ipynb del repositorio
https://github.com/MatiasRamirez7/Deep_Learning (commit 06fb920).
Cada corrección está marcada con una etiqueta [FIX-xx] que se explica en el documento
"diferencias_codigo.docx".

Los notebooks 01_entrenamiento.ipynb y 02_lime.ipynb importan este módulo, de modo que el
modelo que se entrena y el que se explica con LIME se construyen con exactamente la misma función.
"""
from __future__ import annotations

import copy
import json
import os
import platform
import random
from dataclasses import asdict, dataclass, field

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image
from scipy import stats
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, confusion_matrix,
                             f1_score, precision_recall_fscore_support)
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torchvision import models

# ---------------------------------------------------------------------------
# Etiquetas
# ---------------------------------------------------------------------------
# [FIX-09] El diccionario original incluía 'Miedo' sin que el artículo lo mencionara.
# Se mantiene para no perder etiquetas, pero ahora queda documentado y se reporta cuántas
# imágenes lo recibieron (ver load_labels).
EMOTION_TO_VALENCE = {
    'Felicidad': 0, 'Diversión': 0, 'Curiosidad': 0,   # Positive
    'Indiferencia': 1,                                  # Neutral
    'Tristeza': 2, 'Miedo': 2, 'Desagrado': 2,          # Negative
}
# [FIX-17] Nombres de clase únicos (antes se mezclaban Indiferente/Neutral y, en Lime_.ipynb,
# un diccionario de 7 emociones se usaba para titular predicciones de 3 clases).
CLASS_NAMES = ['Positive', 'Neutral', 'Negative']
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406])
IMAGENET_STD = np.array([0.229, 0.224, 0.225])


# ---------------------------------------------------------------------------
# Configuración
# ---------------------------------------------------------------------------
@dataclass
class Config:
    # Rutas: carpeta con las 2.000 imágenes ORIGINALES (sin aumentar) y CSV con sus etiquetas.
    originals_dir: str = 'images_or'
    labels_csv: str = 'etiquetas_originales.csv'
    csv_sep: str | None = None    # None = detecta automáticamente ',' o ';'
    image_col: str | None = None  # None = busca 'image', 'imagen' o 'nombre_imagen'
    label_col: str = 'label'
    output_dir: str = 'resultados'

    pretrained: bool = True
    # [FIX-14] Resolución nativa de cada arquitectura (EfficientNet V2-M fue preentrenada a 480).
    img_size: dict = field(default_factory=lambda: {
        'resnet152': 224, 'resnet50': 224, 'efficientnet_v2_m': 480})
    # [FIX-15] True: redimensiona la imagen completa a un cuadrado (no recorta bordes con texto
    # o logos). False: reproduce Resize(256)+CenterCrop(224) del código original.
    keep_full_image: bool = True
    # Transformaciones de la Tabla I del artículo. [FIX-02] Ahora se aplican al vuelo y SOLO
    # al conjunto de entrenamiento.
    rotation_deg: float = 20.0
    zoom: float = 0.20
    shift: float = 0.10
    horizontal_flip: bool = True
    vertical_flip: bool = True

    # [FIX-01] Partición por imagen original, estratificada por clase.
    test_size: float = 0.15
    val_size: float = 0.15
    test_seed: int = 2024          # el conjunto de test es fijo para todas las semillas
    seeds: tuple = (0, 1, 2, 3, 4)  # [FIX-12] varias semillas → media ± IC95%

    # Optuna
    optuna_trials: int = 15
    optuna_epochs: int = 10
    lr_range: tuple = (1e-4, 1e-1)  # [FIX-13] el código original usaba 1e-5 (el artículo dice 1e-4)
    batch_sizes: tuple = (16, 32, 64, 128, 256)
    optimizers: tuple = ('Adam', 'RMSprop', 'SGD')
    selection_metric: str = 'macro_f1'  # 'macro_f1' o 'accuracy'

    # Entrenamiento final
    final_epochs: int = 45
    patience: int = 10             # [FIX-11] early stopping + se restaura la mejor época
    class_weighted_loss: bool = True  # [FIX-16] manejo del desbalance de clases

    # Linear probe sobre embeddings precalculados (sección 7b)
    embedding_views: int = 4       # vistas aumentadas por imagen, solo para entrenar el probe
    probe_Cs: tuple = (1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 0.1, 0.3, 1.0)
    cv_splits: int = 5
    cv_repeats: int = 5

    # [PERF-02] Procesos que cargan y aumentan imágenes en paralelo con la GPU. Con 0 todo se
    # hacía en el proceso principal y la GPU esperaba. Colab gratis tiene 2 CPU.
    num_workers: int = field(default_factory=lambda: min(8, os.cpu_count() or 1))
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'

    def save(self, path):
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(asdict(self), f, indent=2, ensure_ascii=False)


def set_seed(seed: int):
    """[FIX-12] Fija todas las fuentes de aleatoriedad (el original no fijaba ninguna)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ---------------------------------------------------------------------------
# Datos y partición
# ---------------------------------------------------------------------------
def load_labels(cfg: Config) -> pd.DataFrame:
    """Lee el CSV de etiquetas de las imágenes ORIGINALES (una fila por imagen)."""
    df = pd.read_csv(cfg.labels_csv, sep=cfg.csv_sep, engine='python', encoding='utf-8-sig')
    df.columns = [str(c).replace('\ufeff', '').strip() for c in df.columns]
    image_col = cfg.image_col or next(
        (c for c in ('image', 'imagen', 'nombre_imagen') if c in df.columns), 'image')
    missing_cols = [c for c in (image_col, cfg.label_col) if c not in df.columns]
    if missing_cols:
        raise ValueError(
            f'El CSV no tiene las columnas {missing_cols}. Columnas leídas: {list(df.columns)} '
            f'(separador usado: {cfg.csv_sep!r}). Ajuste csv_sep, image_col y label_col en '
            f'pl.Config o, si acaba de actualizar pipeline.py, reinicie el kernel.')
    df = df.rename(columns={image_col: 'image', cfg.label_col: 'emotion'})
    df['image'] = df['image'].astype(str).str.replace('\ufeff', '').str.strip()
    df['emotion'] = df['emotion'].astype(str).str.replace('\ufeff', '').str.strip()

    unknown = sorted(set(df['emotion']) - set(EMOTION_TO_VALENCE))
    if unknown:
        raise ValueError(f'Emociones no reconocidas en el CSV: {unknown}')
    df['label'] = df['emotion'].map(EMOTION_TO_VALENCE).astype(int)
    df['path'] = df['image'].map(lambda n: os.path.join(cfg.originals_dir, n))

    missing = df.loc[~df['path'].map(os.path.exists), 'image'].tolist()
    if missing:
        raise FileNotFoundError(f'{len(missing)} imágenes del CSV no existen, p.ej. {missing[:5]}')
    if df['image'].duplicated().any():
        raise ValueError('Hay imágenes duplicadas en el CSV.')
    return df.reset_index(drop=True)


def class_distribution(df: pd.DataFrame) -> pd.DataFrame:
    """Distribución por emoción y por valencia (pedida por R1 y R4)."""
    emo = df.groupby(['label', 'emotion']).size().rename('n').reset_index()
    emo['valence'] = emo['label'].map(dict(enumerate(CLASS_NAMES)))
    emo['%'] = (100 * emo['n'] / len(df)).round(1)
    return emo[['valence', 'emotion', 'n', '%']]


def make_splits(df: pd.DataFrame, cfg: Config, seed: int) -> dict:
    """[FIX-01] Partición a nivel de imagen ORIGINAL, estratificada por clase.

    El test es fijo (cfg.test_seed); train/val cambian con cada semilla. Como la augmentation
    se hace al vuelo (FIX-02), ninguna variante de una imagen de val/test llega a entrenamiento.
    """
    trainval, test = train_test_split(df, test_size=cfg.test_size, stratify=df['label'],
                                      random_state=cfg.test_seed)
    rel_val = cfg.val_size / (1 - cfg.test_size)
    train, val = train_test_split(trainval, test_size=rel_val, stratify=trainval['label'],
                                  random_state=seed)
    splits = {'train': train.reset_index(drop=True), 'val': val.reset_index(drop=True),
              'test': test.reset_index(drop=True)}
    assert not (set(splits['train'].image) & set(splits['test'].image))
    assert not (set(splits['train'].image) & set(splits['val'].image))
    return splits


def save_manifest(splits: dict, path: str):
    """Manifiesto de partición (pedido por R4): imagen, etiqueta y partición."""
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    rows = [s.assign(split=name)[['image', 'emotion', 'label', 'split']]
            for name, s in splits.items()]
    pd.concat(rows).to_csv(path, index=False)


def split_distribution(splits: dict) -> pd.DataFrame:
    tab = pd.DataFrame({name: s['label'].value_counts().sort_index() for name, s in splits.items()})
    tab.index = [CLASS_NAMES[i] for i in tab.index]
    return tab


def _resize_op(size: int, cfg: Config):
    if cfg.keep_full_image:
        return T.Resize((size, size), interpolation=T.InterpolationMode.BILINEAR)
    return T.Compose([T.Resize(int(size * 256 / 224)), T.CenterCrop(size)])


def get_transform(arch: str, cfg: Config, train: bool):
    """[FIX-02] Augmentation al vuelo SOLO en entrenamiento (parámetros de la Tabla I)."""
    size = cfg.img_size[arch]
    ops = [_resize_op(size, cfg)]
    if train:
        ops.append(T.RandomAffine(degrees=cfg.rotation_deg, translate=(cfg.shift, cfg.shift),
                                  scale=(1 - cfg.zoom, 1 + cfg.zoom)))
        if cfg.horizontal_flip:
            ops.append(T.RandomHorizontalFlip())
        if cfg.vertical_flip:
            ops.append(T.RandomVerticalFlip())
    ops += [T.ToTensor(), T.Normalize(IMAGENET_MEAN.tolist(), IMAGENET_STD.tolist())]
    return T.Compose(ops)


class ImageDataset(Dataset):
    def __init__(self, df: pd.DataFrame, transform):
        self.paths = df['path'].tolist()
        self.labels = df['label'].tolist()
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        # [FIX-18] convert('RGB'): el original fallaba o se comportaba distinto con imágenes
        # en escala de grises, RGBA o paleta.
        image = Image.open(self.paths[idx]).convert('RGB')
        return self.transform(image), self.labels[idx]


def _seed_worker(worker_id):
    s = torch.initial_seed() % 2 ** 32
    np.random.seed(s)
    random.seed(s)


def make_loader(df, arch, cfg, batch_size, train, seed=0):
    g = torch.Generator()
    g.manual_seed(seed)
    # [PERF-02] persistent_workers: los procesos se crean una vez por loader, no en cada época.
    return DataLoader(ImageDataset(df, get_transform(arch, cfg, train)), batch_size=batch_size,
                      shuffle=train, num_workers=cfg.num_workers, worker_init_fn=_seed_worker,
                      generator=g, pin_memory=cfg.device == 'cuda',
                      persistent_workers=cfg.num_workers > 0)


def prepare_resized_images(cfg: Config, cache_dir: str, min_side: int | None = None,
                           quality: int = 95) -> str:
    """[PERF-01] Copia reducida de cfg.originals_dir en cache_dir, para no decodificar
    imágenes de ~1080×1314 px en cada época. Devuelve cache_dir (asígnelo a cfg.originals_dir).

    Cada imagen se reduce, con su relación de aspecto, hasta que su lado menor mida `min_side`
    (por defecto el mayor tamaño de entrada en cfg.img_size, 480 px). Como después todo se lleva
    a 224 o 480 px, el modelo ve casi lo mismo (no idéntico bit a bit: hay dos remuestreos).
    Las imágenes que ya son más chicas se copian sin tocar. Solo procesa lo que falta, así que
    volver a llamarla es instantáneo. Use un disco local (/content en Colab), no Drive.
    """
    import shutil
    from concurrent.futures import ThreadPoolExecutor

    min_side = min_side or max(cfg.img_size.values())
    os.makedirs(cache_dir, exist_ok=True)
    names = [n for n in os.listdir(cfg.originals_dir)
             if n.lower().endswith(('.jpg', '.jpeg', '.png'))]

    def one(name):
        src, dst = os.path.join(cfg.originals_dir, name), os.path.join(cache_dir, name)
        if os.path.exists(dst):
            return 0
        with Image.open(src) as im:
            w, h = im.size
            if min(w, h) <= min_side:
                shutil.copy2(src, dst)
                return 1
            s = min_side / min(w, h)
            im = im.convert('RGB').resize((round(w * s), round(h * s)), Image.BICUBIC)
            im.save(dst + '.tmp', format='JPEG', quality=quality)
        os.replace(dst + '.tmp', dst)  # una copia a medias nunca queda con el nombre final
        return 1

    with ThreadPoolExecutor(max_workers=max(2, os.cpu_count() or 1)) as ex:
        done = sum(ex.map(one, names))
    print(f'Imágenes reducidas (lado menor {min_side} px) en {cache_dir}: '
          f'{done} nuevas, {len(names) - done} ya estaban')
    return cache_dir


# ---------------------------------------------------------------------------
# Modelos
# ---------------------------------------------------------------------------
ARCH_OF_EXP = {1: 'resnet152', 2: 'resnet152', 3: 'resnet152', 4: 'resnet152', 5: 'resnet152',
               6: 'resnet152', 7: 'efficientnet_v2_m', 8: 'efficientnet_v2_m',
               '4s': 'resnet152', '5s': 'resnet152', '8s': 'efficientnet_v2_m',
               'ft50': 'resnet50'}
# [FIX-05] Experimentos "espaciales": el pooling se aplica DESPUÉS de las 1×1 conv, así estas
# sí operan sobre el mapa de características (en el original operaban sobre un tensor 1×1).
SPATIAL_EXPS = {'4s', '5s', '8s'}
# Baseline de fine-tuning completo (R1: "standard fine-tuned CNN").
FINETUNE_EXPS = {'ft50'}


def _fc_tail(sizes):
    layers = []
    for a, b in zip(sizes[:-1], sizes[1:]):
        layers += [nn.Linear(a, b), nn.ReLU()]
    return layers[:-1]  # sin ReLU tras la última capa


# Cabezas idénticas a las del código original (celda 14), para que la ablación sea comparable.
# [FIX-06] La Tabla II del artículo debe describir ESTAS cabezas: no hay ReLU tras BatchNorm.
HEADS = {
    1: lambda d: nn.Sequential(nn.Flatten(), nn.Linear(d, 3)),
    2: lambda d: nn.Sequential(nn.Flatten(), *_fc_tail([d, 256, 3])),
    3: lambda d: nn.Sequential(nn.Flatten(), *_fc_tail([d, 256, 256, 3])),
    4: lambda d: nn.Sequential(nn.Conv2d(d, 256, 1, bias=False), nn.BatchNorm2d(256), nn.Flatten(),
                               *_fc_tail([256, 256, 128, 3])),
    5: lambda d: nn.Sequential(nn.Conv2d(d, 256, 1, bias=False), nn.BatchNorm2d(256),
                               nn.Conv2d(256, 256, 1, bias=False), nn.BatchNorm2d(256), nn.Flatten(),
                               *_fc_tail([256, 256, 128, 3])),
    6: lambda d: nn.Sequential(nn.Conv2d(d, 256, 1, bias=False), nn.BatchNorm2d(256),
                               nn.Conv2d(256, 256, 1, bias=False), nn.BatchNorm2d(256), nn.Flatten(),
                               *_fc_tail([256, 256, 256, 128, 3])),
    7: lambda d: nn.Sequential(nn.Flatten(), nn.Linear(d, 3)),
    8: lambda d: nn.Sequential(nn.Conv2d(d, 256, 1, bias=False), nn.BatchNorm2d(256), nn.Flatten(),
                               *_fc_tail([256, 256, 128, 3])),
    # Variantes espaciales [FIX-05]
    '4s': lambda d: nn.Sequential(nn.Conv2d(d, 256, 1, bias=False), nn.BatchNorm2d(256), nn.ReLU(),
                                  nn.AdaptiveAvgPool2d(1), nn.Flatten(), *_fc_tail([256, 256, 128, 3])),
    '5s': lambda d: nn.Sequential(nn.Conv2d(d, 256, 1, bias=False), nn.BatchNorm2d(256), nn.ReLU(),
                                  nn.Conv2d(256, 256, 1, bias=False), nn.BatchNorm2d(256), nn.ReLU(),
                                  nn.AdaptiveAvgPool2d(1), nn.Flatten(), *_fc_tail([256, 256, 128, 3])),
    '8s': lambda d: nn.Sequential(nn.Conv2d(d, 256, 1, bias=False), nn.BatchNorm2d(256), nn.ReLU(),
                                  nn.AdaptiveAvgPool2d(1), nn.Flatten(), *_fc_tail([256, 256, 128, 3])),
    'ft50': lambda d: nn.Sequential(nn.Flatten(), nn.Linear(d, 3)),
}


def load_features(arch: str, pretrained: bool):
    """[FIX-04] Carga pesos ImageNet NUEVOS en cada llamada y devuelve solo el extractor
    convolucional (sin avgpool ni clasificador)."""
    if arch == 'resnet152':
        m = models.resnet152(weights=models.ResNet152_Weights.IMAGENET1K_V1 if pretrained else None)
        return nn.Sequential(*list(m.children())[:-2]), 2048
    if arch == 'resnet50':
        m = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None)
        return nn.Sequential(*list(m.children())[:-2]), 2048
    if arch == 'efficientnet_v2_m':
        m = models.efficientnet_v2_m(
            weights=models.EfficientNet_V2_M_Weights.IMAGENET1K_V1 if pretrained else None)
        return m.features, 1280
    raise ValueError(arch)


class BackboneClassifier(nn.Module):
    """Backbone preentrenado + cabeza de clasificación.

    [FIX-03] Con freeze=True el backbone queda realmente congelado:
      - requires_grad=False en todos sus parámetros,
      - siempre en modo eval() (las estadísticas de BatchNorm no cambian aunque se llame a
        model.train()),
      - se ejecuta bajo torch.no_grad().
    """

    def __init__(self, features: nn.Module, head: nn.Module, pool: bool, freeze: bool):
        super().__init__()
        self.features = features
        self.pool = nn.AdaptiveAvgPool2d(1) if pool else nn.Identity()
        self.head = head
        self.freeze = freeze
        if freeze:
            for p in self.features.parameters():
                p.requires_grad = False
            self.features.eval()

    def train(self, mode: bool = True):
        super().train(mode)
        if self.freeze:
            self.features.eval()
        return self

    def forward(self, x):
        if self.freeze:
            with torch.no_grad():
                f = self.features(x)
        else:
            f = self.features(x)
        return self.head(self.pool(f))

    def forward_with_feature_map(self, x):
        """Devuelve el mapa de características (con gradiente) y los logits, para Grad-CAM."""
        with torch.no_grad():
            f = self.features(x)
        f = f.detach().requires_grad_(True)
        return f, self.head(self.pool(f))

    def trainable_state(self):
        src = self.head if self.freeze else self
        return {k: v.detach().cpu().clone() for k, v in src.state_dict().items()}

    def load_trainable_state(self, state):
        (self.head if self.freeze else self).load_state_dict(state)


def build_model(exp, cfg: Config) -> BackboneClassifier:
    """[FIX-04] Construye un modelo NUEVO (backbone recién cargado + cabeza nueva).
    Debe llamarse una vez por experimento, por trial de Optuna y por semilla."""
    arch = ARCH_OF_EXP[exp]
    features, dim = load_features(arch, cfg.pretrained)
    model = BackboneClassifier(features, HEADS[exp](dim), pool=exp not in SPATIAL_EXPS,
                               freeze=exp not in FINETUNE_EXPS)
    return model.to(cfg.device)


def count_params(model: nn.Module):
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total


def backbone_fingerprint(model: BackboneClassifier) -> dict:
    """Copia de pesos Y estadísticas BN del backbone, para verificar que no cambian."""
    return {k: v.detach().cpu().clone() for k, v in model.features.state_dict().items()}


def changed_backbone_tensors(model: BackboneClassifier, fingerprint: dict) -> list:
    return [k for k, v in model.features.state_dict().items()
            if not torch.equal(v.detach().cpu(), fingerprint[k])]


def make_optimizer(name, params, lr):
    if name == 'Adam':
        return torch.optim.Adam(params, lr=lr)
    if name == 'RMSprop':
        return torch.optim.RMSprop(params, lr=lr)
    if name == 'SGD':
        return torch.optim.SGD(params, lr=lr)
    raise ValueError(name)


# ---------------------------------------------------------------------------
# Entrenamiento y evaluación
# ---------------------------------------------------------------------------
def class_weights(train_df: pd.DataFrame) -> torch.Tensor:
    counts = train_df['label'].value_counts().reindex(range(3), fill_value=0).values
    counts = np.maximum(counts, 1)
    return torch.tensor(len(train_df) / (3 * counts), dtype=torch.float32)


@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    ys, logits = [], []
    for x, y in loader:
        logits.append(model(x.to(device, dtype=torch.float32)).cpu())
        ys.append(y)
    logits = torch.cat(logits)
    y = torch.cat(ys)
    return y.numpy(), logits.argmax(1).numpy(), F.softmax(logits, 1).numpy(), \
        F.cross_entropy(logits, y).item()


def compute_metrics(y_true, y_pred) -> dict:
    """[FIX-10] Métricas pedidas por R1/R3/R4: accuracy, balanced accuracy, macro-F1,
    precision/recall/F1 por clase y matriz de confusión (el original solo calculaba F1 ponderado)."""
    p, r, f, s = precision_recall_fscore_support(y_true, y_pred, labels=[0, 1, 2], zero_division=0)
    out = {'accuracy': accuracy_score(y_true, y_pred),
           'balanced_accuracy': balanced_accuracy_score(y_true, y_pred),
           'macro_f1': f1_score(y_true, y_pred, average='macro', labels=[0, 1, 2], zero_division=0),
           'confusion_matrix': confusion_matrix(y_true, y_pred, labels=[0, 1, 2]).tolist()}
    for i, c in enumerate(CLASS_NAMES):
        out[f'precision_{c}'], out[f'recall_{c}'], out[f'f1_{c}'] = p[i], r[i], f[i]
    return out


def fit(model, optimizer, train_loader, val_loader, cfg: Config, epochs: int,
        weights: torch.Tensor | None = None, patience: int | None = None, verbose=True):
    """[FIX-07] Recibe los DataLoaders como argumentos (el original usaba variables globales,
    por lo que el batch size sugerido por Optuna nunca se aplicaba).
    [FIX-11] Guarda la mejor época según cfg.selection_metric y la restaura al final."""
    device = cfg.device
    w = weights.to(device) if weights is not None else None
    best_score, best_epoch, best_state, history = -np.inf, -1, None, []
    for epoch in range(epochs):
        model.train()
        running, n = 0.0, 0
        for x, y in train_loader:
            x, y = x.to(device, dtype=torch.float32), y.to(device, dtype=torch.long)
            loss = F.cross_entropy(model(x), y, weight=w)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running += loss.item() * len(y)
            n += len(y)
        y_val, p_val, _, val_loss = predict(model, val_loader, device)
        m = compute_metrics(y_val, p_val)
        history.append({'epoch': epoch, 'train_loss': running / n, 'val_loss': val_loss,
                        'val_accuracy': m['accuracy'], 'val_macro_f1': m['macro_f1']})
        score = m[cfg.selection_metric]
        if score > best_score:
            best_score, best_epoch, best_state = score, epoch, model.trainable_state()
        if verbose:
            print(f'Epoch {epoch:3d}  train_loss={running / n:.4f}  val_loss={val_loss:.4f}  '
                  f'val_acc={m["accuracy"]:.4f}  val_macroF1={m["macro_f1"]:.4f}')
        if patience is not None and epoch - best_epoch >= patience:
            if verbose:
                print(f'Early stopping en la época {epoch} (mejor: {best_epoch})')
            break
    model.load_trainable_state(best_state)
    return pd.DataFrame(history), best_epoch, best_score


def run_optuna(exp, df: pd.DataFrame, cfg: Config, seed: int | None = None, verbose=False):
    """Búsqueda de hiperparámetros para UN experimento sobre la partición de validación.
    [FIX-04] Cada trial construye un modelo nuevo. [FIX-07] Los loaders con el batch size
    sugerido se pasan a fit()."""
    import optuna
    seed = cfg.seeds[0] if seed is None else seed
    arch = ARCH_OF_EXP[exp]
    splits = make_splits(df, cfg, seed)
    w = class_weights(splits['train']) if cfg.class_weighted_loss else None

    def objective(trial):
        lr = trial.suggest_float('lr', *cfg.lr_range, log=True)
        bs = trial.suggest_categorical('batch_size', list(cfg.batch_sizes))
        opt_name = trial.suggest_categorical('optimizer', list(cfg.optimizers))
        set_seed(seed * 1000 + trial.number)
        model = build_model(exp, cfg)
        train_loader = make_loader(splits['train'], arch, cfg, bs, train=True, seed=seed)
        val_loader = make_loader(splits['val'], arch, cfg, bs, train=False)
        opt = make_optimizer(opt_name, [p for p in model.parameters() if p.requires_grad], lr)
        _, _, best = fit(model, opt, train_loader, val_loader, cfg, cfg.optuna_epochs,
                         weights=w, verbose=verbose)
        del model
        if cfg.device == 'cuda':
            torch.cuda.empty_cache()
        return best

    out = os.path.join(cfg.output_dir, 'optuna')
    os.makedirs(out, exist_ok=True)
    # Estudio persistente en SQLite: si la sesión se corta, se retoma desde los trials completados.
    storage = f'sqlite:///{os.path.abspath(os.path.join(out, "optuna.db"))}'
    study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=seed),
                                study_name=f'exp_{exp}_seed{seed}', storage=storage,
                                load_if_exists=True)
    done = sum(t.state == optuna.trial.TrialState.COMPLETE for t in study.trials)
    if done:
        print(f'Experimento {exp}: {done}/{cfg.optuna_trials} trials ya completados')
    if done < cfg.optuna_trials:
        study.optimize(objective, n_trials=cfg.optuna_trials - done)
    study.trials_dataframe().to_csv(os.path.join(out, f'trials_exp{exp}.csv'), index=False)
    return study


def run_final(exp, params: dict, df: pd.DataFrame, cfg: Config, verbose=False) -> list:
    """Entrenamiento final con los hiperparámetros elegidos, repetido para cada semilla.
    Evalúa UNA vez en el test fijo (imágenes originales, sin aumentar)."""
    arch = ARCH_OF_EXP[exp]
    results = []
    mdir = os.path.join(cfg.output_dir, 'modelos')
    os.makedirs(mdir, exist_ok=True)
    for seed in cfg.seeds:
        # Si esta combinación (exp, semilla, hiperparámetros) ya terminó, se reutiliza.
        res_path = os.path.join(mdir, f'result_exp{exp}_seed{seed}.json')
        if os.path.exists(res_path):
            with open(res_path, encoding='utf-8') as f:
                cached = json.load(f)
            if cached['params'] == params:
                print(f'Experimento {exp}, semilla {seed}: ya entrenado, se carga {res_path}')
                results.append(cached['metrics'])
                continue
        set_seed(seed)
        splits = make_splits(df, cfg, seed)
        save_manifest(splits, os.path.join(cfg.output_dir, 'manifiestos', f'split_seed{seed}.csv'))
        model = build_model(exp, cfg)
        fp = backbone_fingerprint(model) if model.freeze else None
        bs = params['batch_size']
        train_loader = make_loader(splits['train'], arch, cfg, bs, train=True, seed=seed)
        val_loader = make_loader(splits['val'], arch, cfg, bs, train=False)
        test_loader = make_loader(splits['test'], arch, cfg, bs, train=False)
        # [FIX-08] El optimizador recibe solo los parámetros entrenables (la cabeza).
        opt = make_optimizer(params['optimizer'],
                             [p for p in model.parameters() if p.requires_grad], params['lr'])
        w = class_weights(splits['train']) if cfg.class_weighted_loss else None
        hist, best_epoch, _ = fit(model, opt, train_loader, val_loader, cfg, cfg.final_epochs,
                                  weights=w, patience=cfg.patience, verbose=verbose)
        if fp is not None:
            changed = changed_backbone_tensors(model, fp)
            assert not changed, f'El backbone cambió durante el entrenamiento: {changed[:3]}'
        y, p, prob, _ = predict(model, test_loader, cfg.device)
        m = compute_metrics(y, p)
        m.update({'exp': str(exp), 'arch': arch, 'seed': seed, 'best_epoch': best_epoch,
                  'n_train': len(splits['train']), 'n_val': len(splits['val']),
                  'n_test': len(splits['test'])})
        results.append(m)
        torch.save({'exp': str(exp), 'seed': seed, 'params': params,
                    'state': model.trainable_state()},
                   os.path.join(mdir, f'exp{exp}_seed{seed}.pt'))
        hist.to_csv(os.path.join(mdir, f'history_exp{exp}_seed{seed}.csv'), index=False)
        # Se escribe al final y de forma atómica: su existencia indica que la semilla terminó.
        with open(res_path + '.tmp', 'w', encoding='utf-8') as f:
            json.dump({'params': params, 'metrics': m}, f, indent=2, default=float)
        os.replace(res_path + '.tmp', res_path)
        del model
        if cfg.device == 'cuda':
            torch.cuda.empty_cache()
    return results


def load_trained_model(exp, seed, cfg: Config) -> BackboneClassifier:
    ckpt = torch.load(os.path.join(cfg.output_dir, 'modelos', f'exp{exp}_seed{seed}.pt'),
                      map_location=cfg.device)
    model = build_model(exp, cfg)
    model.load_trainable_state(ckpt['state'])
    return model.eval()


def mean_ci(values):
    v = np.asarray(values, dtype=float)
    n = len(v)
    mean = v.mean()
    if n < 2:
        return mean, np.nan, np.nan, np.nan
    sd = v.std(ddof=1)
    half = stats.t.ppf(0.975, n - 1) * sd / np.sqrt(n)
    return mean, sd, mean - half, mean + half


def summarize(results: list,
              metrics=('accuracy', 'balanced_accuracy', 'macro_f1',
                       'f1_Positive', 'f1_Neutral', 'f1_Negative')) -> pd.DataFrame:
    """[FIX-12] Tabla III nueva: media ± DE e IC95% sobre semillas, en el test."""
    df = pd.DataFrame(results)
    rows = []
    for exp, g in df.groupby('exp', sort=False):
        row = {'exp': exp, 'arch': g['arch'].iloc[0], 'n_seeds': len(g)}
        for m in metrics:
            mean, sd, lo, hi = mean_ci(g[m])
            row[m] = mean
            row[f'{m}_sd'] = sd
            row[f'{m}_ci95'] = f'[{lo:.3f}, {hi:.3f}]'
        rows.append(row)
    return pd.DataFrame(rows)


def aggregate_confusion(results: list, exp) -> np.ndarray:
    return sum(np.array(r['confusion_matrix']) for r in results if r['exp'] == str(exp))


def plot_confusion(cm, title='', ax=None, normalize=True):
    import matplotlib.pyplot as plt
    ax = ax or plt.subplots(figsize=(4.5, 4))[1]
    cm = np.asarray(cm, dtype=float)
    shown = cm / cm.sum(1, keepdims=True).clip(min=1) if normalize else cm
    ax.imshow(shown, cmap='Blues', vmin=0, vmax=1 if normalize else None)
    for i in range(3):
        for j in range(3):
            txt = f'{shown[i, j]:.2f}\n({int(cm[i, j])})' if normalize else f'{int(cm[i, j])}'
            ax.text(j, i, txt, ha='center', va='center',
                    color='white' if shown[i, j] > (0.5 if normalize else shown.max() / 2) else 'black')
    ax.set_xticks(range(3), CLASS_NAMES)
    ax.set_yticks(range(3), CLASS_NAMES)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.set_title(title)
    return ax


# ---------------------------------------------------------------------------
# Baselines (R1: "no comparison with any existing method")
# ---------------------------------------------------------------------------
def majority_baseline(df: pd.DataFrame, cfg: Config) -> dict:
    splits = make_splits(df, cfg, cfg.seeds[0])
    majority = splits['train']['label'].mode()[0]
    y = splits['test']['label'].values
    m = compute_metrics(y, np.full_like(y, majority))
    m.update({'exp': 'majority', 'arch': '-', 'seed': cfg.seeds[0],
              'majority_class': CLASS_NAMES[majority]})
    return m


def handcrafted_features(path: str, size: int = 256) -> np.ndarray:
    """Color (HSV), colorido de Hasler-Süsstrunk, brillo, densidad de bordes y entropía."""
    from skimage.color import rgb2gray, rgb2hsv
    from skimage.feature import canny
    from skimage.measure import shannon_entropy
    rgb = np.asarray(Image.open(path).convert('RGB').resize((size, size)), dtype=np.float64) / 255
    hsv = rgb2hsv(rgb)
    feats = []
    for c in range(3):
        h, _ = np.histogram(hsv[..., c], bins=8, range=(0, 1))
        feats += list(h / h.sum())
        feats += [hsv[..., c].mean(), hsv[..., c].std()]
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    rg, yb = r - g, 0.5 * (r + g) - b
    feats.append(np.sqrt(rg.std() ** 2 + yb.std() ** 2) + 0.3 * np.sqrt(rg.mean() ** 2 + yb.mean() ** 2))
    gray = rgb2gray(rgb)
    feats += [gray.mean(), gray.std(), canny(gray).mean(), shannon_entropy(gray)]
    return np.asarray(feats)


def run_svm_baseline(df: pd.DataFrame, cfg: Config, Cs=(0.1, 1, 10, 100)) -> list:
    """Features hand-crafted + SVM RBF, con las MISMAS particiones y semillas que las CNN."""
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC
    feats = {p: handcrafted_features(p) for p in df['path']}
    results = []
    for seed in cfg.seeds:
        s = make_splits(df, cfg, seed)
        X = {k: np.stack([feats[p] for p in v['path']]) for k, v in s.items()}
        y = {k: v['label'].values for k, v in s.items()}
        best = None
        for C in Cs:
            clf = make_pipeline(StandardScaler(), SVC(C=C, gamma='scale', class_weight='balanced',
                                                      random_state=seed))
            clf.fit(X['train'], y['train'])
            score = compute_metrics(y['val'], clf.predict(X['val']))[cfg.selection_metric]
            if best is None or score > best[0]:
                best = (score, C, clf)
        m = compute_metrics(y['test'], best[2].predict(X['test']))
        m.update({'exp': 'svm_handcrafted', 'arch': 'SVM', 'seed': seed, 'C': best[1]})
        results.append(m)
    return results


# ---------------------------------------------------------------------------
# Linear probe sobre embeddings precalculados (sección 7b)
# ---------------------------------------------------------------------------
# Modelos fundacionales (Hugging Face) y, para comparar bajo el MISMO protocolo, los backbones
# torchvision de la ablación (None = se carga con load_features).
EMBEDDING_MODELS = {
    'clip_vitl14': 'openai/clip-vit-large-patch14',
    'siglip_so400m': 'google/siglip-so400m-patch14-384',
    'dinov2_large': 'facebook/dinov2-large',
    'resnet152': None,
    'efficientnet_v2_m': None,
}


def _side(size) -> int | None:
    if size is None:
        return None
    for k in ('height', 'shortest_edge'):
        v = size.get(k) if isinstance(size, dict) else getattr(size, k, None)
        if v:
            return int(v)
    return None


def _load_embedder(key: str, cfg: Config):
    """Devuelve (función batch→embeddings, lado de entrada, media, desviación) para `key`."""
    if EMBEDDING_MODELS[key] is None:
        features, _ = load_features(key, pretrained=True)
        features = features.to(cfg.device).eval()
        return (lambda x: features(x).mean((2, 3))), cfg.img_size[key], IMAGENET_MEAN, IMAGENET_STD

    from transformers import AutoImageProcessor, AutoModel
    name = EMBEDDING_MODELS[key]
    proc = AutoImageProcessor.from_pretrained(name)
    model = AutoModel.from_pretrained(name).to(cfg.device).eval()
    size = _side(getattr(proc, 'crop_size', None)) or _side(proc.size)

    def fn(x):
        if hasattr(model, 'get_image_features'):  # CLIP / SigLIP: espacio imagen-texto
            out = model.get_image_features(pixel_values=x)
            return out if torch.is_tensor(out) else out.pooler_output
        h = model(pixel_values=x).last_hidden_state  # DINOv2: CLS + media de parches
        return torch.cat([h[:, 0], h[:, 1:].mean(1)], 1)

    return fn, size, np.asarray(proc.image_mean), np.asarray(proc.image_std)


def _embedding_transform(size, mean, std, cfg: Config, train: bool):
    """Mismo redimensionado que las CNN [FIX-15]. Las vistas de entrenamiento usan una
    augmentation suave: sin flip vertical ni rotaciones (texto y logos quedan legibles)."""
    ops = [_resize_op(size, cfg)]
    if train:
        ops += [T.RandomResizedCrop(size, scale=(0.8, 1.0), ratio=(0.9, 1.1)),
                T.RandomHorizontalFlip()]
    ops += [T.ToTensor(), T.Normalize(list(mean), list(std))]
    return T.Compose(ops)


@torch.no_grad()
def extract_embeddings(df: pd.DataFrame, key: str, cfg: Config, batch_size: int = 64) -> dict:
    """Embeddings de las imágenes originales ('X', N×D) y de cfg.embedding_views vistas
    aumentadas ('X_aug', V×N×D). El backbone está congelado, así que se calculan UNA vez y se
    guardan en <output_dir>/embeddings/ (se reutilizan si la lista de imágenes coincide)."""
    n_views = cfg.embedding_views
    path = os.path.join(cfg.output_dir, 'embeddings', f'{key}_v{n_views}.npz')
    if os.path.exists(path):
        cached = np.load(path, allow_pickle=False)
        if cached['image'].tolist() == df['image'].tolist():
            print(f'{key}: embeddings cargados de {path}')
            return {'X': cached['X'], 'X_aug': cached['X_aug']}

    fn, size, mean, std = _load_embedder(key, cfg)
    use_amp = cfg.device == 'cuda'

    def run(train):
        loader = DataLoader(ImageDataset(df, _embedding_transform(size, mean, std, cfg, train)),
                            batch_size=batch_size, shuffle=False, num_workers=cfg.num_workers,
                            pin_memory=cfg.device == 'cuda')
        out = []
        for x, _ in loader:
            with torch.autocast('cuda', dtype=torch.float16, enabled=use_amp):
                out.append(fn(x.to(cfg.device)).float().cpu())
        return torch.cat(out).numpy()

    set_seed(0)
    X = run(train=False)
    X_aug = np.stack([run(train=True) for _ in range(n_views)]) if n_views else \
        np.empty((0, *X.shape), dtype=X.dtype)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez(path, X=X, X_aug=X_aug, image=df['image'].to_numpy(dtype=str))
    print(f'{key}: {X.shape[0]} imágenes × {X.shape[1]} dims, {n_views} vistas aumentadas → {path}')
    if cfg.device == 'cuda':
        torch.cuda.empty_cache()
    return {'X': X, 'X_aug': X_aug}


def _probe(C: float, seed: int):
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    return make_pipeline(StandardScaler(),
                         LogisticRegression(C=C, class_weight='balanced', max_iter=5000,
                                            random_state=seed))


def _fit_probe(emb: dict, idx: np.ndarray, y_all: np.ndarray, C: float, seed: int):
    """Entrena con las imágenes `idx` y sus vistas aumentadas (nunca las de otras particiones)."""
    X, y = emb['X'][idx], y_all[idx]
    if len(emb['X_aug']):
        X = np.concatenate([X, emb['X_aug'][:, idx].reshape(-1, X.shape[1])])
        y = np.tile(y, 1 + len(emb['X_aug']))
    return _probe(C, seed).fit(X, y)


def _select_C_cv(X, y, cfg: Config, seed: int) -> float:
    """Elige C por CV interna (5 pliegues, solo originales) con la métrica de selección
    calculada sobre las predicciones fuera de pliegue: más estable que un val con 12 negativas."""
    from sklearn.model_selection import StratifiedKFold, cross_val_predict
    skf = StratifiedKFold(5, shuffle=True, random_state=seed)
    scores = [compute_metrics(y, cross_val_predict(_probe(C, seed), X, y, cv=skf, n_jobs=-1))
              [cfg.selection_metric] for C in cfg.probe_Cs]
    return cfg.probe_Cs[int(np.argmax(scores))]


def run_embedding_final(key: str, emb: dict, df: pd.DataFrame, cfg: Config) -> list:
    """Protocolo de la Tabla III: mismas particiones y semillas que las CNN y el SVM.
    Se entrena en train, se elige C en val y se evalúa UNA vez en el test fijo."""
    pos = {img: i for i, img in enumerate(df['image'])}
    y_all = df['label'].values
    results = []
    for seed in cfg.seeds:
        s = make_splits(df, cfg, seed)
        idx = {k: v['image'].map(pos).values for k, v in s.items()}
        best = None
        for C in cfg.probe_Cs:
            clf = _fit_probe(emb, idx['train'], y_all, C, seed)
            score = compute_metrics(y_all[idx['val']], clf.predict(emb['X'][idx['val']]))[
                cfg.selection_metric]
            if best is None or score > best[0]:
                best = (score, C, clf)
        m = compute_metrics(y_all[idx['test']], best[2].predict(emb['X'][idx['test']]))
        m.update({'exp': f'lp_{key}', 'arch': f'{key}+LR', 'seed': seed, 'C': best[1]})
        results.append(m)
    return results


def run_embedding_cv(key: str, emb: dict, df: pd.DataFrame, cfg: Config) -> list:
    """Validación cruzada estratificada repetida (cfg.cv_repeats × cfg.cv_splits) sobre las
    2.000 imágenes. En cada repetición todas las imágenes (incluidas las 77 Negative) se
    predicen una vez fuera de pliegue; las métricas se calculan por repetición, así que
    summarize() da media ± IC95% sobre repeticiones."""
    from sklearn.model_selection import StratifiedKFold
    path = os.path.join(cfg.output_dir, 'embeddings',
                        f'cv_{key}_v{cfg.embedding_views}_{cfg.cv_repeats}x{cfg.cv_splits}.json')
    if os.path.exists(path):
        print(f'{key}: CV ya calculada, se carga {path}')
        with open(path, encoding='utf-8') as f:
            return json.load(f)
    y_all = df['label'].values
    results = []
    for rep in range(cfg.cv_repeats):
        seed = cfg.test_seed + rep
        pred = np.full(len(y_all), -1)
        Cs = []
        skf = StratifiedKFold(cfg.cv_splits, shuffle=True, random_state=seed)
        for tr, te in skf.split(emb['X'], y_all):
            C = _select_C_cv(emb['X'][tr], y_all[tr], cfg, seed)
            pred[te] = _fit_probe(emb, tr, y_all, C, seed).predict(emb['X'][te])
            Cs.append(C)
        m = compute_metrics(y_all, pred)
        m.update({'exp': f'lp_{key}', 'arch': f'{key}+LR', 'seed': rep, 'C_per_fold': Cs})
        results.append(m)
        print(f'{key}, repetición {rep}: macro-F1={m["macro_f1"]:.3f}  acc={m["accuracy"]:.3f}')
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, default=float)
    return results


def env_info() -> dict:
    """[R1/R3] Hardware y software usados."""
    from importlib.metadata import PackageNotFoundError, version
    info = {'python': platform.python_version(), 'platform': platform.platform(),
            'processor': platform.processor(), 'torch': torch.__version__,
            'cuda_available': torch.cuda.is_available(),
            'cuda': torch.version.cuda, 'cudnn': torch.backends.cudnn.version()}
    for lib in ('torchvision', 'optuna', 'lime', 'scikit-image', 'scikit-learn', 'numpy', 'pandas'):
        try:
            info[lib] = version(lib)
        except PackageNotFoundError:
            info[lib] = None
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        info['gpu'] = props.name
        info['gpu_memory_gb'] = round(props.total_memory / 1024 ** 3, 1)
    try:
        import psutil
        info['ram_gb'] = round(psutil.virtual_memory().total / 1024 ** 3, 1)
    except ImportError:
        pass
    return info


# ---------------------------------------------------------------------------
# LIME
# ---------------------------------------------------------------------------
def load_image_01(path: str, arch: str, cfg: Config) -> np.ndarray:
    """[FIX-19] Imagen en RGB [0, 1] con el MISMO redimensionado que en evaluación.
    La normalización ImageNet se hace dentro de la función de predicción, no antes de LIME."""
    img = _resize_op(cfg.img_size[arch], cfg)(Image.open(path).convert('RGB'))
    return np.asarray(img, dtype=np.float64) / 255.0


def make_predict_fn(model: nn.Module, device: str, batch_size: int = 32):
    model.eval()

    def predict_fn(images: np.ndarray) -> np.ndarray:
        x = (np.asarray(images, dtype=np.float64) - IMAGENET_MEAN) / IMAGENET_STD
        x = torch.from_numpy(x.transpose(0, 3, 1, 2)).float()
        out = []
        with torch.no_grad():
            for i in range(0, len(x), batch_size):
                out.append(F.softmax(model(x[i:i + batch_size].to(device)), 1).cpu())
        return torch.cat(out).numpy()

    return predict_fn


def explain(image01: np.ndarray, predict_fn, kernel_size=6, max_dist=300, ratio=0.3,
            num_samples=1000, random_state=0, batch_size=50) -> dict:
    """[FIX-20] LIME con semilla fija (random_state) en el explicador. hide_color=0 sobre la
    imagen en [0,1] = negro, como dice el artículo (antes era el gris medio de ImageNet)."""
    from lime import lime_image
    from lime.wrappers.scikit_image import SegmentationAlgorithm
    seg = SegmentationAlgorithm('quickshift', kernel_size=kernel_size, max_dist=max_dist,
                                ratio=ratio, random_seed=random_state)
    explainer = lime_image.LimeImageExplainer(random_state=random_state)
    e = explainer.explain_instance(image01, predict_fn, top_labels=1, hide_color=0,
                                   num_samples=num_samples, batch_size=batch_size,
                                   segmentation_fn=seg)
    label = int(e.top_labels[0])
    probs = predict_fn(image01[None])[0]
    # Según la versión de LIME, e.score es un número o un diccionario por etiqueta
    score = e.score[label] if isinstance(e.score, dict) else e.score
    return {'label': label, 'prob': float(probs[label]), 'probs': probs,
            'segments': e.segments, 'weights': dict(e.local_exp[label]),
            'score': float(score), 'n_segments': int(e.segments.max() + 1)}


def weight_map(res: dict) -> np.ndarray:
    """Mapa de pesos LIME por píxel. [FIX-21] Son coeficientes del modelo lineal local,
    no probabilidades."""
    w = np.zeros(res['segments'].max() + 1)
    for k, v in res['weights'].items():
        w[k] = v
    return w[res['segments']]


def topk_segments(weights: dict, k: int) -> list:
    ranked = sorted(((v, s) for s, v in weights.items() if v > 0), reverse=True)
    return [s for _, s in ranked[:k]]


def jaccard_topk(w1: dict, w2: dict, k: int = 5) -> float:
    a, b = set(topk_segments(w1, k)), set(topk_segments(w2, k))
    return len(a & b) / len(a | b) if (a | b) else np.nan


def fragmentation(weights: dict) -> dict:
    """Cuantifica el patrón de 'fragmentación' (R1/R4): entropía normalizada de los pesos
    positivos y número de superpíxeles necesarios para acumular el 50 % del peso positivo."""
    pos = np.sort(np.array([v for v in weights.values() if v > 0]))[::-1]
    if len(pos) == 0:
        return {'entropy_norm': np.nan, 'n50': np.nan, 'n50_frac': np.nan}
    p = pos / pos.sum()
    ent = -(p * np.log(p)).sum() / np.log(len(p)) if len(p) > 1 else 0.0
    n50 = int(np.searchsorted(np.cumsum(p), 0.5) + 1)
    return {'entropy_norm': float(ent), 'n50': n50, 'n50_frac': n50 / len(weights)}


def region_overlap(res: dict, mask: np.ndarray, k: int = 3, n_random: int = 1000, seed: int = 0) -> dict:
    """Solapamiento entre los top-k superpíxeles LIME y una región anotada por humanos
    (producto, cara, cuerpo, texto), con baseline de superpíxeles al azar (R1/R4)."""
    seg, mask = res['segments'], mask.astype(bool)
    if mask.sum() == 0:
        return {}
    top = topk_segments(res['weights'], k)
    if not top:
        return {}

    def _prec_iou(ids):
        sel = np.isin(seg, ids)
        inter = (sel & mask).sum()
        return inter / sel.sum(), inter / (sel | mask).sum()

    prec, iou = _prec_iou(top)
    hit = mask[seg == top[0]].mean() > 0.5
    rng = np.random.default_rng(seed)
    ids = np.unique(seg)
    rand = np.array([_prec_iou(rng.choice(ids, size=min(len(top), len(ids)), replace=False))
                     for _ in range(n_random)])
    hit_rand = np.mean([mask[seg == s].mean() > 0.5 for s in ids])
    return {'hit_top1': bool(hit), 'precision_topk': prec, 'iou_topk': iou,
            'hit_random': hit_rand, 'precision_random': rand[:, 0].mean(),
            'iou_random': rand[:, 1].mean(), 'mask_area': mask.mean()}


def load_mask(path: str, arch: str, cfg: Config) -> np.ndarray:
    size = cfg.img_size[arch]
    img = Image.open(path).convert('L')
    if cfg.keep_full_image:
        img = img.resize((size, size), Image.NEAREST)
    else:
        img = T.Compose([T.Resize(int(size * 256 / 224), interpolation=T.InterpolationMode.NEAREST),
                         T.CenterCrop(size)])(img)
    return np.asarray(img) > 127


def gradcam(model: BackboneClassifier, image01: np.ndarray, target: int, device: str) -> np.ndarray:
    """Grad-CAM sobre la última capa convolucional del backbone (R1: segundo método)."""
    model.eval()
    x = torch.from_numpy(((image01 - IMAGENET_MEAN) / IMAGENET_STD).transpose(2, 0, 1)[None]).float()
    fmap, logits = model.forward_with_feature_map(x.to(device))
    logits[0, target].backward()
    w = fmap.grad.mean(dim=(2, 3), keepdim=True)
    cam = F.relu((w * fmap).sum(1, keepdim=True))
    cam = F.interpolate(cam, size=image01.shape[:2], mode='bilinear', align_corners=False)[0, 0]
    cam = cam.detach().cpu().numpy()
    return (cam - cam.min()) / (cam.max() - cam.min() + 1e-12)


def segment_means(value_map: np.ndarray, segments: np.ndarray) -> np.ndarray:
    counts = np.bincount(segments.ravel())
    sums = np.bincount(segments.ravel(), weights=value_map.ravel())
    return sums / np.maximum(counts, 1)


def spearman_lime_vs_map(res: dict, value_map: np.ndarray) -> float:
    ids = sorted(res['weights'])
    lime_w = np.array([res['weights'][i] for i in ids])
    other = segment_means(value_map, res['segments'])[ids]
    return float(stats.spearmanr(lime_w, other).correlation)


def spearman_between(res_a: dict, res_b: dict) -> float:
    ids = sorted(set(res_a['weights']) & set(res_b['weights']))
    return float(stats.spearmanr([res_a['weights'][i] for i in ids],
                                 [res_b['weights'][i] for i in ids]).correlation)


def randomized_copy(model: BackboneClassifier, n_layers: int, seed: int = 0) -> BackboneClassifier:
    """Prueba de aleatorización de parámetros (Adebayo et al., NeurIPS 2018): reinicializa
    en cascada las n capas con parámetros más cercanas a la salida de la cabeza."""
    torch.manual_seed(seed)
    m = copy.deepcopy(model)
    layers = [l for l in m.head.modules() if hasattr(l, 'reset_parameters') and
              any(True for _ in l.parameters(recurse=False))]
    for layer in layers[::-1][:n_layers]:
        layer.reset_parameters()
        if isinstance(layer, nn.BatchNorm2d):
            layer.reset_running_stats()
    return m.eval()


def n_head_layers(model: BackboneClassifier) -> int:
    return len([l for l in model.head.modules() if hasattr(l, 'reset_parameters') and
                any(True for _ in l.parameters(recurse=False))])


def select_quickshift(images: list, predict_fn, kernel_sizes, max_dists, ratios, num_samples=500,
                      seeds=(0, 1), k=5, min_segments=20, max_segments=150) -> pd.DataFrame:
    """[FIX-22] Selección cuantitativa de Quick Shift sobre VARIAS imágenes de validación:
    fidelidad local (R² de LIME) y estabilidad del top-k entre semillas (antes: juicio visual
    sobre una sola imagen)."""
    rows = []
    for ks in kernel_sizes:
        for md in max_dists:
            for ra in ratios:
                r2, jac, nseg = [], [], []
                for img in images:
                    res = [explain(img, predict_fn, ks, md, ra, num_samples, s) for s in seeds]
                    r2 += [r['score'] for r in res]
                    nseg.append(res[0]['n_segments'])
                    jac += [jaccard_topk(res[0]['weights'], r['weights'], k) for r in res[1:]]
                rows.append({'kernel_size': ks, 'max_dist': md, 'ratio': ra,
                             'mean_r2': np.mean(r2), 'mean_jaccard_topk': np.nanmean(jac),
                             'mean_n_segments': np.mean(nseg)})
    df = pd.DataFrame(rows)
    df['valid'] = df['mean_n_segments'].between(min_segments, max_segments)
    return df.sort_values(['valid', 'mean_jaccard_topk', 'mean_r2'], ascending=False)


# ---------------------------------------------------------------------------
# LIME vs. atención humana estimada (DeepGaze IIE)  — notebook 03
# ---------------------------------------------------------------------------
def save_explanation(res: dict, path: str):
    """Guarda una explicación de explain() (incluye R² y probabilidades, para poder filtrar)."""
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    np.savez_compressed(path, segments=res['segments'],
                        weight_ids=np.array(list(res['weights'].keys())),
                        weight_values=np.array(list(res['weights'].values())),
                        label=res['label'], prob=res['prob'], probs=res['probs'],
                        score=res['score'])


def load_explanation(path: str) -> dict:
    d = np.load(path)
    segments = d['segments']
    return {'label': int(d['label']), 'prob': float(d['prob']), 'probs': d['probs'],
            'segments': segments, 'score': float(d['score']),
            'weights': dict(zip(d['weight_ids'].tolist(), d['weight_values'].tolist())),
            'n_segments': int(segments.max() + 1)}


DEEPGAZE_CENTERBIAS_URL = ('https://github.com/matthias-k/DeepGaze/releases/download/v1.0.0/'
                           'centerbias_mit1003.npy')


def load_deepgaze(cfg: Config, centerbias_path: str):
    """DeepGaze IIE preentrenado (Linardos et al., ICCV 2021) y la plantilla de sesgo central
    de MIT1003 (log-densidad 1024×1024)."""
    import deepgaze_pytorch
    if not os.path.exists(centerbias_path):
        os.makedirs(os.path.dirname(centerbias_path) or '.', exist_ok=True)
        torch.hub.download_url_to_file(DEEPGAZE_CENTERBIAS_URL, centerbias_path)
    model = deepgaze_pytorch.DeepGazeIIE(pretrained=True).to(cfg.device).eval()
    return model, np.load(centerbias_path)


def _to_lime_geometry(density: np.ndarray, arch: str, cfg: Config) -> np.ndarray:
    """Lleva un mapa definido sobre la imagen original a la geometría de load_image_01
    (mismo redimensionado/recorte que ve LIME) y lo renormaliza a suma 1."""
    img = Image.fromarray(density.astype(np.float32))  # modo 'F'
    out = np.clip(np.asarray(_resize_op(cfg.img_size[arch], cfg)(img), dtype=np.float64), 0, None)
    return out / out.sum()


@torch.no_grad()
def deepgaze_maps(path: str, model, centerbias_template: np.ndarray, arch: str, cfg: Config,
                  max_side: int = 1024) -> dict:
    """Tres mapas de densidad (suma 1) en la geometría de LIME:
      - 'deepgaze': predicción completa (contenido + sesgo central de MIT1003);
      - 'deepgaze_content': DeepGaze con sesgo central uniforme (solo contenido de la imagen);
      - 'centerbias': solo el sesgo central (control: ¿basta con 'mirar al centro'?).
    DeepGaze se aplica a la imagen ORIGINAL (con su relación de aspecto, lado mayor = max_side),
    no a la versión cuadrada que ve el clasificador."""
    from scipy.ndimage import zoom
    from scipy.special import logsumexp
    img = Image.open(path).convert('RGB')
    s = max_side / max(img.size)
    img = img.resize((max(1, round(img.width * s)), max(1, round(img.height * s))), Image.BILINEAR)
    x = np.asarray(img)
    h, w = x.shape[:2]
    cb = zoom(centerbias_template, (h / centerbias_template.shape[0], w / centerbias_template.shape[1]),
              order=0, mode='nearest')
    cb -= logsumexp(cb)
    flat = np.full((h, w), -np.log(h * w))
    xt = torch.tensor(x.transpose(2, 0, 1)[None]).to(cfg.device)

    def run(log_cb):
        ld = model(xt, torch.tensor(log_cb[None]).to(cfg.device))
        return np.exp(ld.squeeze().double().cpu().numpy())

    return {'deepgaze': _to_lime_geometry(run(cb), arch, cfg),
            'deepgaze_content': _to_lime_geometry(run(flat), arch, cfg),
            'centerbias': _to_lime_geometry(np.exp(cb), arch, cfg)}


def saliency_alignment(res: dict, sal: np.ndarray, k: int = 5) -> dict:
    """Alineamiento entre una explicación LIME y un mapa de densidad de atención (suma 1).
      - spearman_signed: pesos LIME (con signo) vs saliencia media de cada superpíxel;
      - spearman_abs: |peso| vs saliencia (importancia sin dirección);
      - mass_topk: fracción de la atención que cae en los top-k superpíxeles positivos;
      - area_topk: fracción del área que ocupan esos superpíxeles;
      - enrichment_topk = mass/area (>1: las regiones que usa el modelo atraen más atención
        que una región promedio del mismo tamaño);
      - jaccard_topk: top-k de LIME vs top-k superpíxeles por saliencia media."""
    seg = res['segments']
    ids = sorted(res['weights'])
    w = np.array([res['weights'][i] for i in ids])
    s_mean = segment_means(sal, seg)[ids]
    sel = np.isin(seg, topk_segments(res['weights'], k))
    mass, area = sal[sel].sum(), sel.mean()
    return {'spearman_signed': stats.spearmanr(w, s_mean).correlation,
            'spearman_abs': stats.spearmanr(np.abs(w), s_mean).correlation,
            'mass_topk': mass, 'area_topk': area,
            'enrichment_topk': mass / area if area > 0 else np.nan,
            'jaccard_topk': jaccard_topk(res['weights'], dict(zip(ids, s_mean)), k)}


def permutation_alignment(explanations: dict, sal_maps: dict, k: int = 5, n_perm: int = 50,
                          seed: int = 0) -> pd.DataFrame:
    """Baseline de permutación: la explicación de cada imagen contra el mapa DeepGaze de OTRAS
    imágenes. Si el alineamiento no baja, lo medido es una regularidad de composición de los
    anuncios (p. ej. el producto al centro), no una coincidencia propia de cada imagen."""
    rng = np.random.default_rng(seed)
    names = list(explanations)
    rows = []
    for i, name in enumerate(names):
        others = rng.choice([j for j in range(len(names)) if j != i],
                            size=min(n_perm, len(names) - 1), replace=False)
        m = pd.DataFrame([saliency_alignment(explanations[name], sal_maps[names[j]], k)
                          for j in others]).mean()
        rows.append({'image': name, **m.to_dict()})
    return pd.DataFrame(rows)
