from __future__ import annotations
import csv
import json
import os
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional


def utc_timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def save_json(results: List[Dict[str, Any]], out_dir: str, filename: Optional[str] = None) -> str:
    ensure_dir(out_dir)
    if filename is None:
        filename = f"results_{utc_timestamp()}.json"
    out_path = os.path.join(out_dir, filename)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    return out_path


def save_csv(results: List[Dict[str, Any]], out_dir: str, filename: Optional[str] = None) -> str:
    ensure_dir(out_dir)
    if filename is None:
        filename = f"results_{utc_timestamp()}.csv"
    out_path = os.path.join(out_dir, filename)

    # union all keys to avoid missing columns
    fieldnames = []
    seen = set()
    for r in results:
        for k in r.keys():
            if k not in seen:
                seen.add(k)
                fieldnames.append(k)

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            writer.writerow(r)

    return out_path