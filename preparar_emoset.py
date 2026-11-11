#!/usr/bin/env python3
"""Prepara imágenes NEGATIVAS de EmoSet-118K (Yang et al., ICCV 2023) como datos extra de
entrenamiento. No requiere pedir permiso a nadie: es un zip público en Dropbox.

EmoSet etiqueta 118.102 imágenes con las 8 emociones del modelo de Mikels. Cada etiqueta
la dieron 10 personas y solo se conserva si más de 7 coinciden, así que es bastante más
limpia que una etiqueta individual. Las 4 emociones negativas y su equivalente en el
proyecto:

    anger   (10.660 imágenes) -> Desagrado
    disgust (10.666)          -> Desagrado
    fear    (13.453)          -> Miedo
    sadness (12.676)          -> Tristeza

Uso:
  1. Bajar EmoSet-118K.zip (enlace en https://github.com/JingyuanYY/EmoSet) o pasar
     la URL con --url (se fuerza dl=1).
  2. python preparar_emoset.py --source EmoSet-118K.zip --per-class 300

Salida:
  images_extra/emoset_<emoción>_<n>.jpg   (lado mayor reducido a --max-side, 480 por defecto)
  etiquetas_externas_negativas.csv        columnas: image,label,source,emoset_emotion

Límites que conviene tener presentes:
  * Son fotos de bancos de imágenes (Openverse, Pexels, Pixabay, RawPixel), NO anuncios.
    Sirven para enseñar qué es "negativo" visualmente, pero hay cambio de dominio.
  * Úselas solo para ENTRENAR. Validación y test deben seguir siendo imágenes del proyecto.
  * Revise los términos de uso de las imágenes antes de publicar nada derivado de ellas;
    el repositorio de EmoSet es MIT, pero eso no dice nada de las fotos.
"""
import argparse
import csv
import os
import random
import shutil
import sys
import tempfile
import urllib.request
import zipfile
from pathlib import Path

from PIL import Image

NEGATIVAS = {"anger": "Desagrado", "disgust": "Desagrado", "fear": "Miedo", "sadness": "Tristeza"}
EXT = {".jpg", ".jpeg", ".png"}


def descargar(url, destino):
    url = url.replace("dl=0", "dl=1")
    if "dl=1" not in url:
        url += ("&" if "?" in url else "?") + "dl=1"
    print(f"Descargando {url[:80]}...")
    with urllib.request.urlopen(url) as r, open(destino, "wb") as f:
        total = int(r.headers.get("Content-Length") or 0)
        hecho = 0
        while chunk := r.read(1 << 20):
            f.write(chunk)
            hecho += len(chunk)
            if total:
                print(f"\r  {hecho / total:5.1%}", end="", flush=True)
    print()


def listar(source):
    """Devuelve {emoción: [(nombre_en_origen, abrir())]} para las emociones negativas."""
    por_emocion = {e: [] for e in NEGATIVAS}
    if zipfile.is_zipfile(source):
        z = zipfile.ZipFile(source)
        for n in z.namelist():
            p = Path(n)
            if p.suffix.lower() in EXT and "image" in p.parts and p.parent.name in NEGATIVAS:
                por_emocion[p.parent.name].append((n, lambda n=n: z.open(n)))
    else:
        raiz = Path(source)
        for e in NEGATIVAS:
            carpeta = next((c for c in (raiz / "image" / e, raiz / e) if c.is_dir()), None)
            if carpeta:
                for f in sorted(carpeta.iterdir()):
                    if f.suffix.lower() in EXT:
                        por_emocion[e].append((str(f), lambda f=f: open(f, "rb")))
    return por_emocion


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", help="EmoSet-118K.zip o carpeta ya descomprimida")
    ap.add_argument("--url", help="URL del zip (Dropbox); alternativa a --source")
    ap.add_argument("--out-images", default="images_extra")
    ap.add_argument("--out-csv", default="etiquetas_externas_negativas.csv")
    ap.add_argument("--per-class", type=int, default=300,
                    help="imágenes por emoción de EmoSet (4 emociones); 0 = todas")
    ap.add_argument("--max-side", type=int, default=480, help="lado mayor en px (0 = sin reducir)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dry-run", action="store_true", help="solo contar, no escribir imágenes")
    a = ap.parse_args()

    tmp = None
    source = a.source
    if not source:
        if not a.url:
            sys.exit("Indique --source (zip o carpeta) o --url.")
        tmp = tempfile.mkdtemp()
        source = os.path.join(tmp, "EmoSet-118K.zip")
        descargar(a.url, source)
    if not os.path.exists(source):
        sys.exit(f"No existe {source}")

    por_emocion = listar(source)
    print("Imágenes negativas encontradas:", {e: len(v) for e, v in por_emocion.items()})
    if not any(por_emocion.values()):
        sys.exit("No encontré carpetas image/<emoción>/. ¿Es EmoSet-118K?")

    rng = random.Random(a.seed)
    filas, out = [], Path(a.out_images)
    for emocion, items in por_emocion.items():
        items = list(items)
        rng.shuffle(items)
        if a.per_class:
            items = items[: a.per_class]
        if a.dry_run:
            print(f"  {emocion}: usaría {len(items)}")
            continue
        out.mkdir(parents=True, exist_ok=True)
        malas = 0
        for k, (origen, abrir) in enumerate(items):
            nombre = f"emoset_{emocion}_{k:05d}.jpg"
            try:
                with abrir() as fh, Image.open(fh) as im:
                    im = im.convert("RGB")
                    if a.max_side and max(im.size) > a.max_side:
                        im.thumbnail((a.max_side, a.max_side), Image.LANCZOS)
                    im.save(out / nombre, quality=92)
            except Exception as ex:  # imagen corrupta: se omite
                malas += 1
                continue
            filas.append((nombre, NEGATIVAS[emocion], "emoset", emocion))
        print(f"  {emocion}: {len(items) - malas} guardadas" + (f" ({malas} corruptas)" if malas else ""))

    if not a.dry_run:
        with open(a.out_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["image", "label", "source", "emoset_emotion"])
            w.writerows(filas)
        print(f"\n{len(filas)} imágenes -> {out}/  y  {a.out_csv}")
    if tmp:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
