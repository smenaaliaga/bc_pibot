from __future__ import annotations

import argparse
from pathlib import Path


def should_process(path: Path, exts: set[str]) -> bool:
    return path.is_file() and path.suffix.lower() in exts


def remove_lines_with_token(path: Path, token: str, *, apply_changes: bool) -> int:
    try:
        original = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return 0

    lines = original.splitlines(keepends=True)
    filtered = [line for line in lines if token not in line]
    removed = len(lines) - len(filtered)

    if removed > 0 and apply_changes:
        path.write_text("".join(filtered), encoding="utf-8")

    return removed


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Remove all lines that contain a given token from text files."
    )
    parser.add_argument(
        "root",
        nargs="?",
        default=".",
        help="Root directory to scan (default: current directory).",
    )
    parser.add_argument(
        "--token",
        default="gasto_value",
        help="Token to search for in each line (default: gasto_value).",
    )
    parser.add_argument(
        "--ext",
        nargs="*",
        default=[".py", ".md", ".json", ".yaml", ".yml", ".toml", ".txt", ".sql"],
        help="File extensions to process.",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Apply changes. Without this flag it only reports matches.",
    )

    args = parser.parse_args()

    root = Path(args.root).resolve()
    exts = {ext.lower() if ext.startswith(".") else f".{ext.lower()}" for ext in args.ext}

    total_files = 0
    total_removed = 0

    for path in root.rglob("*"):
        if ".git" in path.parts or "__pycache__" in path.parts:
            continue
        if not should_process(path, exts):
            continue

        removed = remove_lines_with_token(path, args.token, apply_changes=args.apply)
        if removed > 0:
            total_files += 1
            total_removed += removed
            rel = path.relative_to(root)
            mode = "UPDATED" if args.apply else "MATCH"
            print(f"[{mode}] {rel} -> removed_lines={removed}")

    if args.apply:
        print(f"Done. Updated files: {total_files}, removed lines: {total_removed}")
    else:
        print(
            "Dry-run complete. "
            f"Files with matches: {total_files}, lines that would be removed: {total_removed}. "
            "Run again with --apply to write changes."
        )


if __name__ == "__main__":
    main()
