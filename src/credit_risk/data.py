"""Download and validate a public, real-world credit-default dataset."""
from __future__ import annotations

from pathlib import Path
import hashlib
import requests
import pandas as pd

# UCI Machine Learning Repository: Default of Credit Card Clients (Taiwan).
# Official dataset page: https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients
# The raw XLS file is hosted by UCI. This is observed historical data, not generated data.
DATA_URL = "https://archive.ics.uci.edu/static/public/350/default+of+credit+card+clients.zip"
TARGET_CANDIDATES = {"default payment next month", "default.payment.next.month"}


def download_dataset(
    cache_path: str | Path = "data/raw/default_of_credit_card_clients.xls",
    refresh: bool = False,
) -> tuple[pd.DataFrame, str]:
    """Download the original UCI archive, extract the Excel dataset, and return a checksum.

    Requires network access only when no cached raw file exists or refresh=True.
    """
    import io
    import zipfile

    path = Path(cache_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if refresh or not path.exists():
        response = requests.get(DATA_URL, timeout=60)
        response.raise_for_status()
        with zipfile.ZipFile(io.BytesIO(response.content)) as archive:
            candidates = [n for n in archive.namelist() if n.lower().endswith((".xls", ".xlsx"))]
            if not candidates:
                raise RuntimeError(f"No Excel dataset found in UCI archive: {archive.namelist()}")
            raw = archive.read(candidates[0])
        path.write_bytes(raw)
    payload = path.read_bytes()
    digest = hashlib.sha256(payload).hexdigest()
    # UCI workbook includes a first row with column labels and an ID column.
    df = pd.read_excel(path, header=1)
    df.columns = [str(c).strip() for c in df.columns]
    target = next((c for c in df.columns if c.lower() in TARGET_CANDIDATES), None)
    if target is None:
        # Some mirrors/export versions normalize spaces and punctuation differently.
        target = next((c for c in df.columns if "default" in c.lower() and "next" in c.lower()), None)
    if target is None:
        raise ValueError(f"Could not identify target column. Available columns: {df.columns.tolist()}")
    df = df.rename(columns={target: "default"})
    if "ID" in df.columns:
        df = df.drop(columns=["ID"])
    elif "id" in df.columns:
        df = df.drop(columns=["id"])
    df["default"] = pd.to_numeric(df["default"], errors="raise").astype(int)
    if not set(df["default"].unique()).issubset({0, 1}):
        raise ValueError("Target must be binary (0/1).")
    if len(df) < 1000:
        raise ValueError(f"Dataset unexpectedly small: {len(df)} rows.")
    return df, digest


def validate_and_clean(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Minimal, transparent cleaning; preserves unknown categories and avoids target leakage."""
    if "default" not in df:
        raise ValueError("Expected a 'default' target column.")
    y = df["default"].astype(int).copy()
    X = df.drop(columns=["default"]).copy()
    # Remove duplicate records only if exact feature rows are duplicated; report them upstream.
    # Keep rows by default to avoid silent changes to the original sample.
    X = X.replace([float("inf"), float("-inf")], pd.NA)
    return X, y
