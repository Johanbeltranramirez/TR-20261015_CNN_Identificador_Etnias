"""Prepara datos reales UTKFace en /data (interfaz en español).

Descarga el espejo funcional de Hugging Face (parquet con imágenes + edad,
género y etnia) y extrae una muestra balanceada por etnia con nombres en
formato UTKFace: [edad]_[genero]_[etnia]_[fecha].jpg
  género: 0=hombre, 1=mujer · etnia: 0=Blanca, 1=Negra, 2=Asiática, 3=India, 4=Otras

Genera:
  data/raw/UTKFace/*.jpg      (conjunto de trabajo local, ignorado por git)
  data/muestra/UTKFace/*.jpg  (muestra incluida en el repo para demo sin internet)

Uso:
  python -m src.extraer_muestra                  # usa el parquet local si existe
  python -m src.extraer_muestra --descargar      # descarga el parquet y luego extrae
  python -m src.extraer_muestra --cuota-raw 400 --cuota-muestra 25
"""
from __future__ import annotations

import argparse
import io
import random
import sys
import urllib.request
from pathlib import Path

URL_PARQUET = (
    "https://huggingface.co/datasets/vtsouval/utkface-cropped/resolve/main/"
    "data/train-00000-of-00001.parquet"
)
PARQUET = Path("data/raw/utkface-cropped.parquet")
DEST_RAW = Path("data/raw/UTKFace")
DEST_MUESTRA = Path("data/muestra/UTKFace")
SEMILLA = 42
NOMBRES = ["Blanca", "Negra", "Asiática", "India", "Otras"]


def descargar_parquet(url=URL_PARQUET, destino=PARQUET):
    destino.parent.mkdir(parents=True, exist_ok=True)
    print(f"Descargando espejo UTKFace (~111 MB) desde:\n  {url}")
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req) as resp, open(destino, "wb") as f:
        total = int(resp.headers.get("Content-Length", 0))
        hecho = 0
        while True:
            bloque = resp.read(1024 * 1024)
            if not bloque:
                break
            f.write(bloque)
            hecho += len(bloque)
            if total:
                print(f"  {hecho/1e6:.1f}/{total/1e6:.1f} MB ({100*hecho/total:.0f} %)", end="\r")
    print(f"\nGuardado en {destino} ({destino.stat().st_size/1e6:.1f} MB)")


def extraer(parquet=PARQUET, cuota_raw=140, cuota_muestra=25, semilla=SEMILLA):
    import pyarrow.parquet as pq
    from PIL import Image

    random.seed(semilla)
    DEST_RAW.mkdir(parents=True, exist_ok=True)
    DEST_MUESTRA.mkdir(parents=True, exist_ok=True)

    pf = pq.ParquetFile(parquet)
    print(f"Lectura de {parquet} ({pf.metadata.num_rows} filas)")
    por_etnia = {r: [] for r in range(5)}
    for lote in pf.iter_batches(batch_size=2000, columns=["age", "gender", "race"]):
        base = sum(len(v) for v in por_etnia.values())
        for i, (edad, genero, etnia) in enumerate(zip(
                lote.column("age").to_pylist(),
                lote.column("gender").to_pylist(),
                lote.column("race").to_pylist())):
            if etnia is None or not 0 <= etnia <= 4:
                continue
            por_etnia[etnia].append((base + i, int(edad), int(genero)))
    print("Disponibles por etnia:", {NOMBRES[r]: len(v) for r, v in por_etnia.items()})

    objetivo = {}
    for r in range(5):
        if len(por_etnia[r]) < cuota_raw:
            raise SystemExit(f"Etnia {NOMBRES[r]}: solo hay {len(por_etnia[r])} filas y se pedían {cuota_raw}")
        for j, (fila, edad, genero) in enumerate(random.sample(por_etnia[r], cuota_raw)):
            objetivo[fila] = (r, j, edad, genero)

    n_raw = n_muestra = vistas = 0
    for lote in pf.iter_batches(batch_size=1000, columns=["image"]):
        for k, img in enumerate(lote.column("image").to_pylist()):
            fila = vistas + k
            if fila not in objetivo:
                continue
            r, j, edad, genero = objetivo[fila]
            datos = img["bytes"] if isinstance(img, dict) else img
            foto = Image.open(io.BytesIO(datos)).convert("RGB")
            nombre = f"{edad}_{genero}_{r}_20240101000000{j:05d}.jpg"
            foto.save(DEST_RAW / nombre, "JPEG", quality=90)
            n_raw += 1
            if j < cuota_muestra:
                foto.save(DEST_MUESTRA / nombre, "JPEG", quality=90)
                n_muestra += 1
        vistas += lote.num_rows
    print(f"Guardadas: trabajo={n_raw} en {DEST_RAW} | muestra={n_muestra} en {DEST_MUESTRA}")


def main():
    ap = argparse.ArgumentParser(description="Descarga y extrae UTKFace real en /data.")
    ap.add_argument("--descargar", action="store_true", help="Descargar el parquet antes de extraer")
    ap.add_argument("--cuota-raw", type=int, default=140, help="Imágenes por etnia para trabajo")
    ap.add_argument("--cuota-muestra", type=int, default=25, help="Imágenes por etnia para la muestra del repo")
    args = ap.parse_args()
    if args.descargar or not PARQUET.exists():
        if PARQUET.exists() and not args.descargar:
            pass
        else:
            descargar_parquet()
    extraer(cuota_raw=args.cuota_raw, cuota_muestra=args.cuota_muestra)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
