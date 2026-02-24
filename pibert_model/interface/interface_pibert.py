#!/usr/bin/env python3
"""Interface sencilla para consultar el modelo PIBERT y devolver un JSON con las cabeceras."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, Optional

from predict_cli import Predictor

ROOT = Path(__file__).resolve().parents[1]
_DEFAULT_MODEL_DIR = ROOT / "model_out" / "pibert"
_DEFAULT_DATA_DIR = ROOT / "data"


def _patch_training_args(model_dir: Path) -> None:
    training_args_path = model_dir / "training_args.bin"
    if not training_args_path.exists():
        return
    import torch

    args = torch.load(training_args_path, weights_only=False)
    if hasattr(args, "data_dir"):
        args.data_dir = str(_DEFAULT_DATA_DIR)
    if hasattr(args, "task") and not getattr(args, "task", None):
        args.task = "pibimacecv5"
    torch.save(args, training_args_path)


def predict_query(text: str, model_dir: Optional[str] = None, use_cuda: bool = False) -> Dict[str, Any]:
    """Ejecuta una consulta contra PIBERT y devuelve el resultado como dict."""
    model_path = Path(model_dir) if model_dir else _DEFAULT_MODEL_DIR
    _patch_training_args(model_path)
    predictor = Predictor(str(model_path), no_cuda=not use_cuda)
    return predictor.predict(text)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Interface PIBERT: recibe una consulta y retorna JSON")
    parser.add_argument("--text", "-t", required=True, help="Consulta a evaluar")
    parser.add_argument("--model_dir", default=str(_DEFAULT_MODEL_DIR), help="Directorio del modelo")
    parser.add_argument("--use-cuda", action="store_true", help="Intentar usar CUDA si está disponible")
    return parser.parse_args()


def main():
    args = parse_args()
    result = predict_query(args.text, model_dir=args.model_dir, use_cuda=args.use_cuda)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
