"""
Clone a Hugging Face repository into the project, under model/<category>/<name>.

Usage:
    uv run python model/scripts/clone_hf_model.py <repo_id> [category]

Examples:
    # Clone BETO tokenizer locally under model/tokenizers/
    uv run python model/scripts/clone_hf_model.py dccuchile/bert-base-spanish-wwm-cased tokenizers

    # Clone a fine-tuned model weights under model/weights/
    uv run python model/scripts/clone_hf_model.py my-org/pibot_model_beto weights

Notes:
    - category: one of ["weights", "tokenizers"]. Default: "weights".
    - The destination folder name will be the last segment of repo_id.
"""

from pathlib import Path
import argparse
import sys

try:
    from huggingface_hub import snapshot_download
except Exception as e:
    print("Error: huggingface_hub not installed. Please add it to your environment.")
    raise


def clone(repo_id: str, category: str = "weights") -> Path:
    if category not in {"weights", "tokenizers"}:
        raise ValueError(f"Invalid category: {category}. Use 'weights' or 'tokenizers'.")
    project_root = Path(__file__).resolve().parents[2]
    dest_root = project_root / "model" / category
    dest_root.mkdir(parents=True, exist_ok=True)
    name = repo_id.split("/")[-1]
    dest_dir = dest_root / name
    dest_dir.mkdir(parents=True, exist_ok=True)

    print(f"Cloning {repo_id} -> {dest_dir}")
    snapshot_download(repo_id, local_dir=str(dest_dir))
    print("Done.")
    return dest_dir


def main(argv=None):
    parser = argparse.ArgumentParser(description="Clone HF repo into model/<category>/")
    parser.add_argument("repo_id", help="HuggingFace repo id, e.g. dccuchile/bert-base-spanish-wwm-cased")
    parser.add_argument("category", nargs="?", default="weights", help="Destination category: weights or tokenizers")
    args = parser.parse_args(argv)
    clone(args.repo_id, args.category)


if __name__ == "__main__":
    main()
