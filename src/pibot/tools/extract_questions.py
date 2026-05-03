#!/usr/bin/env python
"""Extracts question=...' entries from a PIBot log file.

Usage (Windows cmd):
    python tools\extract_questions.py --logfile "E:\bc_pibot\logs\pibot_20251128_202413.log" --output extracted_questions.txt

Options:
    --logfile PATH         Path to the log file to parse (required)
    --output PATH          Output file (txt). Defaults to <logfile>.questions.txt
    --csv PATH             If provided, also write a CSV with header 'question'
    --unique               Deduplicate questions (preserve first occurrence order)
    --encoding ENC         File encoding (default utf-8)

The script looks for patterns like:
    question='¿Qué actividades económicas impulsaron/contribuyeron/incidieron al PIB del trimestre 4 2025?'

Notes:
- Uses a conservative regex that stops at the next single quote; if the log format ever embeds escaped single quotes inside a question, refine the pattern.
- Lines without the pattern are ignored.
"""
from __future__ import annotations
import argparse
import os
import re
import sys
from typing import List

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Extract questions from a PIBot log file.")
    p.add_argument("--logfile", required=True, help="Path to log file")
    p.add_argument("--output", help="Output text file (one question per line)")
    p.add_argument("--csv", help="Optional CSV output path")
    p.add_argument("--unique", action="store_true", help="Deduplicate questions")
    p.add_argument("--encoding", default="utf-8", help="File encoding (default utf-8)")
    return p.parse_args()

QUESTION_RE = re.compile(r"question='([^']*)'")

def extract_questions(text: str) -> List[str]:
    # Find all occurrences per entire file content
    return QUESTION_RE.findall(text)

def deduplicate(items: List[str]) -> List[str]:
    seen = set()
    out = []
    for q in items:
        if q not in seen:
            seen.add(q)
            out.append(q)
    return out

def main() -> int:
    args = parse_args()
    log_path = args.logfile
    if not os.path.isfile(log_path):
        print(f"[ERROR] Log file not found: {log_path}", file=sys.stderr)
        return 2
    try:
        with open(log_path, "r", encoding=args.encoding, errors="replace") as f:
            content = f.read()
    except Exception as e:
        print(f"[ERROR] Failed to read file: {e}", file=sys.stderr)
        return 3
    questions = extract_questions(content)
    if args.unique:
        questions = deduplicate(questions)
    if not questions:
        print("[WARN] No questions found with pattern question='...'", file=sys.stderr)
    # Decide output path
    out_path = args.output or (log_path + ".questions.txt")
    try:
        with open(out_path, "w", encoding="utf-8") as f:
            for q in questions:
                f.write(q.strip() + "\n")
    except Exception as e:
        print(f"[ERROR] Failed writing output file: {e}", file=sys.stderr)
        return 4
    if args.csv:
        try:
            with open(args.csv, "w", encoding="utf-8") as fcsv:
                fcsv.write("question\n")
                for q in questions:
                    # Escape double quotes if needed
                    cleaned = q.replace("\"", "\\\"")
                    fcsv.write(f'"{cleaned}"\n')
        except Exception as e:
            print(f"[ERROR] Failed writing CSV: {e}", file=sys.stderr)
            return 5
    print(f"[INFO] Extracted {len(questions)} question(s) -> {out_path}")
    if args.csv:
        print(f"[INFO] CSV written -> {args.csv}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
