"""Utility: check that a features CSV contains only numeric entries (except 'Label').

Usage: python3 code/check_features_numeric.py enhanced_features.csv
Exit code 0 -> numeric-only (success). Non-zero -> found non-numeric value.
"""
import sys
import pandas as pd


def is_numeric_csv(path):
    df = pd.read_csv(path)
    # If 'Label' exists, skip it; otherwise assume last column is Label
    cols = [c for c in df.columns if c != 'Label']
    try:
        for c in cols:
            # Attempt to coerce entire column to numeric, raising on error
            pd.to_numeric(df[c], errors='raise')
    except Exception as e:
        print(f"Non-numeric data detected in column {c}: {e}")
        return False
    return True


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('Usage: check_features_numeric.py <enhanced_features.csv>')
        sys.exit(2)
    path = sys.argv[1]
    ok = is_numeric_csv(path)
    if not ok:
        sys.exit(1)
    print('All feature columns numeric.')
    sys.exit(0)
