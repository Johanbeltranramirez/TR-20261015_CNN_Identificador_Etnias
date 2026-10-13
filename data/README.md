# Datos del proyecto (UTKFace)

## Formato de archivos
Cada imagen sigue el formato original UTKFace:

```
[edad]_[genero]_[etnia]_[fecha].jpg
```

- `edad`: 0–116 · `genero`: 0=hombre, 1=mujer
- `etnia`: 0=Blanca, 1=Negra, 2=Asiática, 3=India, 4=Otras

## Estructura de `data/`

```
data/
├── raw/UTKFace/*.jpg            # conjunto de trabajo (4000 imágenes, 800 por etnia)
├── raw/utkface-cropped.parquet  # espejo original descargado (23.704 filas, ~111 MB)
├── muestra/UTKFace/*.jpg        # muestra incluida en el repo (125 imágenes, 25 por etnia)
└── README.md
```

`data/raw/` está ignorado por git (es grande y se regenera). `data/muestra/`
sí se incluye en el repo para demostraciones sin internet.

## Fuente funcional (el enlace original de Drive está caído)

El enlace original `drive.google.com/.../0BxYys69jI14kU0I1YUQyY1ZDRUE`
devuelve **404**. Fuente verificada que sí funciona:

- **Hugging Face `vtsouval/utkface-cropped`**: UTKFace recortado y limpio en un
  solo parquet (23.704 filas con imagen + edad + género + etnia).
  https://huggingface.co/datasets/vtsouval/utkface-cropped

Distribución real del espejo: Blanca 10.077, Negra 4.526, Asiática 3.434,
India 3.975, Otras 1.692 (desbalance típico del dataset).

## Regenerar los archivos

```bash
pip install -r requirements.txt   # incluye pyarrow + pillow para la extracción
python -m src.extraer_muestra --descargar --cuota-raw 800
# Opciones: --cuota-raw 800 --cuota-muestra 25
```

Esto descarga el parquet a `data/raw/` y extrae una muestra **balanceada y
aleatoria** (semilla 42) con nombres en formato UTKFace.

## Licencia
Las fotos son de UTKFace (Zhang et al., CVPR 2017), solo investigación no
comercial; © de sus dueños originales. Este repo incluye únicamente una
muestra pequeña con fines educativos.
