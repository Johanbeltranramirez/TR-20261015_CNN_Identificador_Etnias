# Muestra UTKFace incluida en el repo

125 imágenes reales (25 por etnia) en formato `[edad]_[genero]_[etnia]_[fecha].jpg`,
extraídas del espejo `vtsouval/utkface-cropped` (ver `data/README.md`).

Sirve para explorar, visualizar y probar el entrenamiento sin descargar nada:

```bash
python -m src.train --data-dir data/muestra/UTKFace --epochs 5 --model simple
streamlit run app/app.py
```

Uso educativo; solo investigación no comercial.
