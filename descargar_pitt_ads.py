#!/usr/bin/env python3
"""Descarga los anuncios NEGATIVOS del dataset de anuncios de la Universidad de Pittsburgh
(Hussain et al., CVPR 2017, https://people.cs.pitt.edu/~kovashka/ads/) y les asigna etiqueta.

El dataset no trae valencia: cada anuncio fue visto por 3 a 5 personas y cada una marcó una o
más de 30 emociones ("sad", "afraid", "cheerful"...). Este script:

  1. Lee esas anotaciones (annotations_images.zip → Sentiments.json y Sentiments_List.txt).
  2. Se queda con los anuncios en que una fracción suficiente de las personas marcó al menos
     una emoción negativa (por defecto: afraid, alarmed, angry, disturbed, pessimistic, sad).
  3. Extrae SOLO esas imágenes de los zips del dataset (subfolder-0.zip ... subfolder-10.zip).
  4. Escribe un CSV con el mismo formato que etiquetas_originales.csv (image,label), donde
     label es Miedo, Tristeza o Desagrado (las tres emociones que el proyecto agrupa como
     Negative), más columnas para auditar: cuántas personas, qué fracción y qué marcaron.

Las URLs de las imágenes las entregan los autores por correo (ver la página del dataset).
Con ellas:

  # solo contar, sin descargar imágenes
  python descargar_pitt_ads.py --dry-run

  # descargar desde las URLs ({i} se reemplaza por 0..10)
  python descargar_pitt_ads.py --images "https://.../subfolder-{i}.zip"

  # o desde zips / carpetas ya descargados
  python descargar_pitt_ads.py --images /ruta/a/los/zips

IMPORTANTE: use estas imágenes solo para ENTRENAR. Validación y test deben seguir siendo
anuncios del dataset propio; si no, los resultados dejan de ser comparables.
"""
import argparse
import ast
import csv
import io
import json
import os
import re
import shutil
import sys
import tempfile
import urllib.request
import zipfile
from collections import Counter

ANNOTATIONS_URL = 'https://people.cs.pitt.edu/~kovashka/ads/annotations_images.zip'
N_SUBFOLDERS = 11

# Las 30 emociones del dataset, en el orden de Sentiments_List.txt (id = posición, desde 1).
# Solo se usa si no se puede leer ese archivo; si se lee, manda el archivo.
SENTIMENTS = ['active', 'afraid', 'alarmed', 'alert', 'amazed', 'amused', 'angry', 'calm',
              'cheerful', 'confident', 'conscious', 'creative', 'disturbed', 'eager', 'educated',
              'emotional', 'empathetic', 'fashionable', 'feminine', 'grateful', 'inspired',
              'jealous', 'loving', 'manly', 'persuaded', 'pessimistic', 'proud', 'sad', 'thrifty',
              'youthful']

# Emoción negativa de Pittsburgh → emoción del proyecto (las tres se agrupan como Negative).
# "jealous" queda fuera a propósito: en publicidad suele ser deseo por el producto.
NEGATIVE_TO_LABEL = {'afraid': 'Miedo', 'alarmed': 'Miedo', 'sad': 'Tristeza',
                     'pessimistic': 'Tristeza', 'disturbed': 'Desagrado', 'angry': 'Desagrado'}


def log(*a):
    print(*a, file=sys.stderr, flush=True)


def fetch(src: str, dest_dir: str) -> str:
    """Devuelve una ruta local para `src` (ruta o URL); las URLs se descargan a dest_dir."""
    if not re.match(r'^https?://', src):
        return src
    path = os.path.join(dest_dir, os.path.basename(src.split('?')[0]) or 'descarga.zip')
    log(f'Descargando {src}')
    req = urllib.request.Request(src, headers={'User-Agent': 'descargar_pitt_ads/1.0'})
    with urllib.request.urlopen(req, timeout=120) as r, open(path + '.part', 'wb') as f:
        total, done, shown = int(r.headers.get('Content-Length') or 0), 0, 0
        while chunk := r.read(1 << 20):
            f.write(chunk)
            done += len(chunk)
            if done - shown >= 200 << 20:
                shown = done
                log(f'  {done >> 20} de {total >> 20} MB' if total else f'  {done >> 20} MB')
    os.replace(path + '.part', path)
    return path


def read_annotations(src: str, tmp: str):
    """(votos, id→emoción). votos: {'3/40803.jpg': [['18', '19'], ['19'], ...]}."""
    path = fetch(src, tmp)
    texts = {}
    if os.path.isdir(path):
        for root, _, files in os.walk(path):
            for n in files:
                if n.lower() in ('sentiments.json', 'sentiments_list.txt'):
                    with open(os.path.join(root, n), encoding='utf-8', errors='replace') as f:
                        texts[n.lower()] = f.read()
    else:
        with zipfile.ZipFile(path) as z:
            for n in z.namelist():
                base = os.path.basename(n).lower()
                if base in ('sentiments.json', 'sentiments_list.txt') and 'video' not in n.lower():
                    texts[base] = z.read(n).decode('utf-8', errors='replace')
    if 'sentiments.json' not in texts:
        sys.exit(f'No encontré Sentiments.json en {src}')
    try:
        votes = json.loads(texts['sentiments.json'])
    except json.JSONDecodeError:
        votes = ast.literal_eval(texts['sentiments.json'])

    names = {}
    for line in texts.get('sentiments_list.txt', '').splitlines():
        m = re.match(r'\s*(\d+)\W+([A-Za-z]+)', line)
        if m:
            names[int(m.group(1))] = m.group(2).lower()
    builtin = {i + 1: s for i, s in enumerate(SENTIMENTS)}
    if set(names.values()) != set(SENTIMENTS):
        if names:
            log('AVISO: no pude interpretar Sentiments_List.txt; uso la lista incluida en el '
                'script. Revise que los ids coincidan con ese archivo.')
        names = builtin
    elif names != builtin:
        log('AVISO: los ids de Sentiments_List.txt difieren de la lista del script; uso los '
            'del archivo.')
    return votes, names


def select_negative(votes: dict, names: dict, negative: set, min_annotators: int,
                    min_agreement: float):
    """Anuncios en que >= min_agreement de las personas marcó alguna emoción negativa."""
    rows = []
    for key, per_annotator in votes.items():
        ann = [[names.get(int(s)) for s in a if str(s).strip().lstrip('-').isdigit()]
               for a in per_annotator]
        ann = [a for a in ann if a]
        if len(ann) < min_annotators:
            continue
        frac = sum(any(s in negative for s in a) for a in ann) / len(ann)
        if frac < min_agreement:
            continue
        counts = Counter(s for a in ann for s in set(a))
        top = max((s for s in counts if s in negative), key=lambda s: (counts[s], s))
        sub, name = key.replace('\\', '/').split('/')[-2:]
        rows.append({'image': f'pitt_{sub}_{name}', 'label': NEGATIVE_TO_LABEL.get(top, 'Desagrado'),
                     'pitt_path': f'{sub}/{name}', 'annotators': len(ann),
                     'negative_fraction': round(frac, 3), 'top_negative': top,
                     'votes': ' '.join(f'{s}:{c}' for s, c in counts.most_common())})
    return sorted(rows, key=lambda r: r['pitt_path'])


def image_sources(spec: str):
    """Fuentes de imágenes a partir de --images: plantilla con {i}, carpeta o archivo zip."""
    if '{i}' in spec:
        return [spec.format(i=i) for i in range(N_SUBFOLDERS)]
    if os.path.isdir(spec):
        zips = sorted(os.path.join(spec, n) for n in os.listdir(spec) if n.lower().endswith('.zip'))
        return zips or [spec]
    return [spec]


def valid_image(data: bytes) -> bool:
    try:
        from PIL import Image
    except ImportError:
        return len(data) > 0
    try:
        with Image.open(io.BytesIO(data)) as im:
            im.verify()
        return True
    except Exception:
        return False


def extract(sources, rows, out_dir: str, tmp: str, keep_zips: bool):
    wanted = {r['pitt_path']: r['image'] for r in rows}
    os.makedirs(out_dir, exist_ok=True)
    found = {p for p, n in wanted.items() if os.path.exists(os.path.join(out_dir, n))}
    if found:
        log(f'{len(found)} imágenes ya estaban en {out_dir}')

    def save(key, data):
        if key in wanted and key not in found and valid_image(data):
            with open(os.path.join(out_dir, wanted[key]), 'wb') as f:
                f.write(data)
            found.add(key)

    for src in sources:
        if len(found) == len(wanted):
            break
        try:
            path = fetch(src, tmp)
        except Exception as e:
            log(f'  no se pudo descargar {src}: {e}')
            continue
        if os.path.isdir(path):
            for root, _, files in os.walk(path):
                for n in files:
                    key = f'{os.path.basename(root)}/{n}'
                    if key in wanted and key not in found:
                        with open(os.path.join(root, n), 'rb') as f:
                            save(key, f.read())
        else:
            with zipfile.ZipFile(path) as z:
                for n in z.namelist():
                    key = '/'.join(n.split('/')[-2:])
                    if key in wanted and key not in found:
                        save(key, z.read(n))
            if path != src and not keep_zips:
                os.remove(path)
        log(f'  {os.path.basename(src)}: {len(found)} de {len(wanted)} imágenes')
    return found


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0],
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument('--annotations', default=ANNOTATIONS_URL,
                    help='annotations_images.zip (URL, archivo o carpeta descomprimida)')
    ap.add_argument('--images', help='plantilla de URL con {i} (0..10), carpeta con los zips o '
                                     'con las imágenes, o un zip. Sin esto solo se cuenta.')
    ap.add_argument('--out-images', default='images_pitt')
    ap.add_argument('--out-csv', default='etiquetas_pitt_negativas.csv')
    ap.add_argument('--negative', nargs='+', default=sorted(NEGATIVE_TO_LABEL),
                    help='emociones de Pittsburgh que cuentan como negativas')
    ap.add_argument('--min-annotators', type=int, default=3)
    ap.add_argument('--min-agreement', type=float, default=0.6,
                    help='fracción mínima de personas que marcó alguna emoción negativa')
    ap.add_argument('--limit', type=int, help='máximo de imágenes (las de mayor acuerdo primero)')
    ap.add_argument('--keep-zips', action='store_true', help='no borrar los zips descargados')
    ap.add_argument('--dry-run', action='store_true', help='solo contar; no tocar imágenes')
    args = ap.parse_args()

    unknown = set(args.negative) - set(SENTIMENTS)
    if unknown:
        sys.exit(f'Emociones desconocidas: {sorted(unknown)}. Válidas: {SENTIMENTS}')

    with tempfile.TemporaryDirectory() as tmp:
        votes, names = read_annotations(args.annotations, tmp)
        rows = select_negative(votes, names, set(args.negative), args.min_annotators,
                               args.min_agreement)
        if args.limit:
            rows = sorted(rows, key=lambda r: (-r['negative_fraction'], -r['annotators'],
                                               r['pitt_path']))[:args.limit]
        log(f'{len(votes)} anuncios con emociones anotadas; {len(rows)} cumplen el criterio '
            f'(>= {args.min_annotators} personas, >= {args.min_agreement:.0%} marcó alguna de: '
            f'{", ".join(sorted(args.negative))})')
        log('Por etiqueta:', dict(Counter(r['label'] for r in rows)))
        log('Por emoción más votada:', dict(Counter(r['top_negative'] for r in rows)))

        if args.dry_run or not args.images:
            if not args.dry_run:
                log('Sin --images no se descargan imágenes. Pida las URLs a los autores '
                    '(https://people.cs.pitt.edu/~kovashka/ads/) y vuelva a ejecutar.')
            return
        found = extract(image_sources(args.images), rows, args.out_images, tmp, args.keep_zips)

    rows = [r for r in rows if r['pitt_path'] in found]
    with open(args.out_csv + '.tmp', 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]) if rows else ['image', 'label'])
        w.writeheader()
        w.writerows(rows)
    shutil.move(args.out_csv + '.tmp', args.out_csv)
    log(f'{len(rows)} imágenes en {args.out_images}/ y sus etiquetas en {args.out_csv}')


if __name__ == '__main__':
    main()
