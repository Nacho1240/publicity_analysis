# Fuentes de datos

El proyecto trabaja con tres fuentes además de las 2.000 imágenes publicitarias propias
(`images_or/` + `etiquetas_originales.csv`).

| Fuente | Qué es | Etiquetas | Dónde está | Estado |
|---|---|---|---|---|
| **OASIS** (Kurdi et al., 2017) | 900 fotos naturales | Valencia y activación 1–7, ~100 personas por imagen | `oasis/` (ya en el repo) | Lista. La usa el notebook 04 |
| **EmoSet-118K** (Yang et al., ICCV 2023) | 118.102 fotos de bancos de imágenes | 8 emociones de Mikels; 10 personas por imagen, se conserva con más de 7 de acuerdo | Zip público en Dropbox, enlace en <https://github.com/JingyuanYY/EmoSet> | Hay que bajar el zip. `preparar_emoset.py` |
| **Pitt Ads** (Hussain et al., CVPR 2017) | 64.832 anuncios (30.340 con emociones) | 30 emociones, 3–5 personas por anuncio | <https://people.cs.pitt.edu/~kovashka/ads/> | Las imágenes se piden por correo a los autores. `descargar_pitt_ads.py` |

Los papers de las tres están en `papers/`.

## Equivalencia con las clases del proyecto

| Clase del proyecto | Emociones propias | EmoSet (Mikels) | Pitt Ads |
|---|---|---|---|
| Positive | Felicidad, Diversión, Curiosidad | amusement, awe, contentment, excitement | sin definir todavía |
| Neutral | Indiferencia | no tiene | no tiene |
| Negative | Desagrado, Miedo, Tristeza | anger y disgust → Desagrado, fear → Miedo, sadness → Tristeza | disturbed y angry → Desagrado, afraid y alarmed → Miedo, sad y pessimistic → Tristeza |

OASIS no tiene emociones: su valencia 1–7 se lleva a 3 clases como en el notebook 04
(1–3 Negative, 4 Neutral, 5–7 Positive).

Ninguna fuente externa aporta la clase Neutral con el mismo significado que "Indiferencia".

## Scripts

Los dos se ejecutan desde la raíz del repositorio y por ahora extraen solo la clase Negative,
que es la que falta (77 de 2.000 imágenes propias).

```
python datos_externos/preparar_emoset.py --source EmoSet-118K.zip --per-class 300
python datos_externos/descargar_pitt_ads.py --dry-run
python datos_externos/descargar_pitt_ads.py --images /ruta/a/los/zips
```

Cada uno deja `datos_externos/<fuente>/images/` (no se sube al repo) y un CSV de etiquetas con
el mismo formato que `etiquetas_originales.csv` (`image,label`) más columnas de auditoría.

**Estado de las pruebas:** ambos scripts se probaron solo con datos simulados, no con los
datasets reales. La URL de las anotaciones de Pitt que trae el script por defecto no está
verificada; si falla, pase el zip con `--annotations`.

## Reglas de uso

- Las imágenes externas van **solo al entrenamiento**. Validación y test siguen siendo
  imágenes propias; si no, los resultados dejan de ser comparables con la Tabla III.
- `pipeline.py` todavía no tiene una opción para sumar imágenes solo al train.
- EmoSet son fotos de bancos de imágenes, no anuncios: hay cambio de dominio.
- Revisar los términos de uso de cada fuente antes de publicar material derivado.
