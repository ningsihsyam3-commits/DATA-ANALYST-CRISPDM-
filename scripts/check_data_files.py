#!/usr/bin/env python3
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"
RAW_DATA_FILE = DATA_DIR / "raw" / "hospital_data_raw.csv"
PROCESSED_DATA_FILE = DATA_DIR / "processed" / "hospital_data_cleaned.csv"
EVALUATION_REPORT = RESULTS_DIR / "evaluation" / "evaluation_report.json"
MODEL_METRICS = RESULTS_DIR / "evaluation" / "model_metrics_isolation_forest.json"

REQUIRED_FILES = {
    "Raw dataset": RAW_DATA_FILE,
    "Processed dataset": PROCESSED_DATA_FILE,
    "Evaluation report": EVALUATION_REPORT,
    "Model metrics": MODEL_METRICS,
}


def format_path(path: Path) -> str:
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def main() -> int:
    missing = {name: path for name, path in REQUIRED_FILES.items() if not path.exists()}

    if not missing:
        print("✅ All required files are present.")
        return 0

    print("❌ Missing required files:")
    for name, path in missing.items():
        print(f"- {name}: {format_path(path)}")

    print("\nSuggested next steps:")
    if RAW_DATA_FILE.exists():
        print("1. Run the data cleaning pipeline to generate processed data:")
        print("   python src/preparation/data_cleaning.py")
    else:
        print(f"1. Add the raw dataset at: {format_path(RAW_DATA_FILE)}")
        print("2. Then run the data cleaning pipeline:")
        print("   python src/preparation/data_cleaning.py")

    print("3. If you have model evaluation output, place it in:")
    print(f"   {format_path(EVALUATION_REPORT)}")
    print(f"   {format_path(MODEL_METRICS)}")

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
