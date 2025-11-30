import pandas as pd
import os

def load_questions(file_path: str, column_name: str = "Question") -> list[str]:
    """
    Reads questions from an Excel file.
    
    Args:
        file_path (str): Path to the .xlsx file.
        column_name (str): Name of the column containing questions. Defaults to "Question".
        
    Returns:
        list[str]: A list of questions found in the file. Returns empty list on error or if file missing.
    """
    if not os.path.exists(file_path):
        print(f"Warning: File not found at {file_path}")
        return []
        
    try:
        # First try reading with header
        df = pd.read_excel(file_path)
        
        if column_name in df.columns:
            return df[column_name].dropna().astype(str).tolist()
        
        # If column not found, check if it might be a headerless file
        # We'll assume the first column is the questions if the specific column isn't found
        print(f"Column '{column_name}' not found. Using the first column as questions.")
        
        # Reload without header to get the first row as data too
        df = pd.read_excel(file_path, header=None)
        if not df.empty:
            return df.iloc[:, 0].dropna().astype(str).tolist()
            
        return []
    except Exception as e:
        print(f"Error reading Excel file: {e}")
        return []
