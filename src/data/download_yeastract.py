#!/usr/bin/env python3
"""
Download the Saccharomyces cerevisiae documented gene regulatory network
from YEASTRACT as an edge list and (optionally) a TF×gene binary matrix.

Outputs:
  - edges_yeastract_documented.csv (columns: tf, gene, reg_type, extra)
  - matrix_yeastract_documented.csv (rows: genes, cols: TFs, 0/1)

References:
  - YEASTRACT Web Services (documented regulations): https://yeastract.com/services/regulations/documented
  - Regulation Matrix help (binary 0/1 meaning): https://yeastract.com/help_regmatrix.php
"""
import argparse
import csv
import io
import sys
from pathlib import Path

import pandas as pd
import requests

BASE = "https://yeastract.com/services/regulations/documented"


def fetch_raw() -> bytes:
    """Fetch documented regulations from YEASTRACT."""
    session = requests.Session()
    session.headers.update({"User-Agent": "yeastract-downloader/1.0"})
    candidates = [
        f"{BASE}?format=json",
        BASE,
        f"{BASE}?format=tsv",
    ]

    last_err = None
    for url in candidates:
        try:
            r = session.get(url, timeout=60)
            r.raise_for_status()
            if r.content and len(r.content) > 0:
                return r.content
        except Exception as e:
            last_err = e

    raise SystemExit(f"[ERROR] Could not fetch YEASTRACT documented data. Last error: {last_err}")


def parse_edges(raw: bytes):
    """Parse JSON or delimited text into a list of edges."""
    # Try JSON
    try:
        import json
        data = json.loads(raw.decode("utf-8"))
        edges = []
        for row in data:
            tf = row.get("tf") or row.get("TF") or row.get("protein") or row.get("regulator")
            gene = row.get("orf") or row.get("gene") or row.get("target") or row.get("locus")
            edges.append({
                "tf": str(tf).upper() if tf else None,
                "gene": str(gene).upper() if gene else None,
                "reg_type": "documented",
                "extra": {k: v for k, v in row.items() if k not in {"tf","TF","protein","regulator","orf","gene","target","locus"}}
            })
        return [e for e in edges if e["tf"] and e["gene"]]
    except Exception:
        pass

    # Fallback: TSV/CSV
    text = raw.decode("utf-8", errors="replace")
    sample = text.strip().splitlines()[:2]
    dialect = "excel-tab" if ("\t" in sample[0]) else "excel"
    reader = csv.reader(io.StringIO(text), dialect=dialect)
    rows = list(reader)

    header = None
    first = rows[0] if rows else []
    if any(h.lower() in {"tf", "transcription_factor", "protein"} for h in map(str.lower, first)):
        header = [h.strip() for h in first]
        data_rows = rows[1:]
    else:
        data_rows = rows

    edges = []
    for fields in data_rows:
        if not fields:
            continue
        tf = fields[0] if len(fields) > 0 else None
        gene = fields[1] if len(fields) > 1 else None
        edges.append({
            "tf": str(tf).upper() if tf else None,
            "gene": str(gene).upper() if gene else None,
            "reg_type": "documented",
            "extra": {}
        })
    return [e for e in edges if e["tf"] and e["gene"]]


def save_edges(edges, out_csv: Path):
    df = pd.DataFrame([{
        "tf": e["tf"],
        "gene": e["gene"],
        "reg_type": e["reg_type"],
        **e["extra"]
    } for e in edges]).drop_duplicates()

    df = df[(df["tf"].notna()) & (df["gene"].notna())]
    df.to_csv(out_csv, index=False)
    return df


def to_matrix(df_edges: pd.DataFrame, out_csv: Path):
    """Build a 0/1 matrix (genes x TFs)."""
    df = df_edges[["tf", "gene"]].dropna().drop_duplicates()
    df["value"] = 1
    mat = df.pivot_table(index="gene", columns="tf", values="value", aggfunc="max", fill_value=0)
    mat.sort_index(axis=0, inplace=True)
    mat.sort_index(axis=1, inplace=True)
    mat.to_csv(out_csv)
    return mat


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--outdir",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "data" / "raw" / "yeastract",
        help="Directory to write outputs",
    )
    ap.add_argument("--matrix", action="store_true",
                    help="Also write a binary TF×gene matrix CSV")
    args = ap.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)

    print("[INFO] Fetching YEASTRACT documented regulations")
    raw = fetch_raw()
    edges = parse_edges(raw)
    if not edges:
        sys.exit("[ERROR] No edges parsed.")

    edges_path = args.outdir / "edges_yeastract_documented.csv"
    df_edges = save_edges(edges, edges_path)
    print(f"[OK] Wrote edge list: {edges_path}  (n={len(df_edges)})")

    if args.matrix:
        mat_path = args.outdir / "matrix_yeastract_documented.csv"
        to_matrix(df_edges, mat_path)
        print(f"[OK] Wrote matrix:    {mat_path}  (shape={df_edges['gene'].nunique()}×{df_edges['tf'].nunique()})")


if __name__ == "__main__":
    main()
