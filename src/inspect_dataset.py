
from pathlib import Path
import pandas as pd

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "raw" / "hackathon_preli_dataset.xlsx"

def inspect_dataset():
    print("=" * 60)
    print("DENGUEGUARD AI - DATASET INSPECTION")
    print("=" * 60)

    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

    # Identify available Excel sheets
    excel_file = pd.ExcelFile(DATA_PATH)

    print("\nAvailable sheets:")
    for sheet in excel_file.sheet_names:
        print("-", sheet)

    # Read the first sheet initially
    df = pd.read_excel(DATA_PATH, sheet_name=excel_file.sheet_names[0])

    print("\nDataset dimensions:")
    print("Rows:", df.shape[0])
    print("Columns:", df.shape[1])

    print("\nColumn names:")
    for column in df.columns:
        print("-", column)

    print("\nFirst 10 records:")
    print(df.head(10).to_string())

    print("\nData types:")
    print(df.dtypes.to_string())

    print("\nMissing values:")
    print(df.isnull().sum().to_string())

    print("\nStatistical summary:")
    print(df.describe(include="all").to_string())

    print("\nDataset inspection completed.")

if __name__ == "__main__":
    inspect_dataset()
