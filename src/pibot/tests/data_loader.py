import pandas as pd
import os

def load_questions(file_path: str, column_name: str = "Question") -> list[str]:
    """
    Reads questions from an Excel (.xlsx) or CSV file.
    """
    if not os.path.exists(file_path):
        print(f"Warning: File not found at {file_path}")
        return []

    def _read_df(path: str):
        if path.lower().endswith(".csv"):
            return pd.read_csv(path)
        return pd.read_excel(path)

    try:
        df = _read_df(file_path)

        if column_name in df.columns:
            return df[column_name].dropna().astype(str).tolist()

        # If column not found, try first column
        print(f"Column '{column_name}' not found. Using the first column as questions.")
        df = _read_df(file_path)
        if not df.empty:
            return df.iloc[:, 0].dropna().astype(str).tolist()
        return []
    except Exception as e:
        print(f"Error reading file: {e}")
        return []
