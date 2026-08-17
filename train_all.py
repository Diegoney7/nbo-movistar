"""
Entrena todos los modelos en el orden correcto, desde cero.

El orden importa y no es intercambiable:

  1. segmentation    — independiente. Produce los artefactos de segmentación que
                       el motor NBO necesita para clasificar al cliente.
  2. acceptance      — produce model_features / model_cat_features /
                       model_categories, que definen el contrato de features
                       compartido por todo el pipeline.
  3. rebate          — CONSUME ese contrato del paso 2. Ejecutarlo antes falla.

Uso:
    python train_all.py
"""
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src'))

from models.segmentation import train_segmentation
from models.acceptance_model import train_acceptance_model
from models.rebate_model import train_rebate_model

PASOS = [
    ('Segmentación de clientes (KMeans)', train_segmentation),
    ('Modelo de aceptación (LightGBM + calibración)', train_acceptance_model),
    ('Modelo de rebate (prior de motivos)', train_rebate_model),
]


def main():
    inicio = time.time()
    for i, (nombre, fn) in enumerate(PASOS, start=1):
        print(f"\n{'=' * 78}")
        print(f"[{i}/{len(PASOS)}] {nombre}")
        print('=' * 78)
        t0 = time.time()
        fn()
        print(f"--> completado en {time.time() - t0:.1f}s")

    print(f"\n{'=' * 78}")
    print(f"Pipeline completo en {time.time() - inicio:.1f}s.")
    print("Levantar la aplicación con:  streamlit run app/streamlit_app.py")
    print('=' * 78)


if __name__ == "__main__":
    main()
