"""
Script para clonar (descargar) un modelo de Hugging Face a model/out/{modelo}

Uso:
    python clone_hf_model.py nombre-del-modelo [ruta_destino]

Ejemplo:
    python clone_hf_model.py dccuchile/bert-base-spanish-wwm-cased model/out/beto
"""
import sys
from huggingface_hub import snapshot_download

if len(sys.argv) < 2:
    print("Uso: python clone_hf_model.py <nombre-del-modelo> [ruta_destino]")
    sys.exit(1)

modelo = sys.argv[1]
if len(sys.argv) > 2:
    ruta_destino = sys.argv[2]
else:
    ruta_destino = f"model/out/{modelo.split('/')[-1]}"

print(f"Descargando modelo '{modelo}' a '{ruta_destino}' ...")
snapshot_download(repo_id=modelo, local_dir=ruta_destino, local_dir_use_symlinks=False)
print("Descarga completada.")
