from pathlib import Path
import re

def clean_pdf_txt(
    input_path: str,
    output_path: str,
    *,
    keep_double_newlines: bool = True,
    remove_hyphen_breaks: bool = True
) -> None:
    """
    Clean text extracted from PDF:
    - Remove form feed chars.
    - Collapse multiple spaces and tabs.
    - Merge broken lines inside paragraphs.
    - Optionally fix hyphenated line breaks.
    """

    raw = Path(input_path).read_text(encoding="utf-8", errors="ignore")

    # 1) Normalize line endings
    text = raw.replace("\r\n", "\n").replace("\r", "\n")

    # 2) Replace form feeds (page breaks) with blank lines
    text = text.replace("\f", "\n\n")

    # 3) Optionally fix hyphenated line breaks:
    #    "ejemplo-\ncontinuacion" -> "ejemplo continuacion"
    if remove_hyphen_breaks:
        text = re.sub(r"-\s*\n\s*", " ", text)

    # 4) Split in blocks separated by blank lines (paragraphs)
    #    Uno o mas saltos de linea en blanco delimitan parrafos
    blocks = re.split(r"\n\s*\n+", text)

    cleaned_blocks = []
    for block in blocks:
        # Partir el bloque en lineas
        lines = block.split("\n")

        # Quitar espacios al inicio/fin de cada linea
        lines = [ln.strip() for ln in lines if ln.strip() != ""]

        if not lines:
            continue

        # Unir lineas del mismo parrafo con espacio
        joined = " ".join(lines)

        # Colapsar multiples espacios y tabs en un solo espacio
        joined = re.sub(r"[ \t]+", " ", joined)

        cleaned_blocks.append(joined)

    # 5) Volver a unir parrafos:
    #    - con doble salto de linea entre ellos (mas legible)
    #    - o todos seguidos si no quieres saltos
    if keep_double_newlines:
        cleaned = "\n\n".join(cleaned_blocks)
    else:
        cleaned = " ".join(cleaned_blocks)

    # 6) Remover líneas que quedaron vacías (solo saltos o espacios)
    cleaned = "\n".join(ln for ln in cleaned.splitlines() if ln.strip())

    # 7) Guardar resultado
    Path(output_path).write_text(cleaned, encoding="utf-8")


if __name__ == "__main__":
    # Ejemplo de uso
    input_file = r"E:\bc_pibot\docker\postgres\docs\Cuentas_Nacionales_metodos_fuentes_ref18_all_20251203_200609.txt"       # tu archivo leido desde el pdf
    output_file = r"E:\bc_pibot\docker\postgres\docs\Cuentas_Nacionales_metodos_fuentes_ref18_all_20251203_200609_limpio.txt"  # archivo limpio de salida

    clean_pdf_txt(input_file, output_file)
    print(f"Texto limpio guardado en: {output_file}")
