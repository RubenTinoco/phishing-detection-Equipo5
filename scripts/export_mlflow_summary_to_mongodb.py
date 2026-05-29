from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mongodb_utils import (
    get_collection,
    get_mongo_client,
    insert_many_documents,
    load_mongodb_config_from_env,
    ping_mongodb,
    sanitize_document,
)


REPORTS_DIR = ROOT / "reports"
MLFLOW_SUMMARY_PATH = REPORTS_DIR / "mlflow_tracking_summary.csv"
BUSINESS_VALUE_SUMMARY_PATH = REPORTS_DIR / "business_value_summary.csv"
BUSINESS_VALUE_SCENARIOS_PATH = REPORTS_DIR / "business_value_scenarios.csv"
EXPORT_SUMMARY_PATH = REPORTS_DIR / "mongodb_export_summary.csv"


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def rel_path(path):
    return Path(path).relative_to(ROOT).as_posix()


def read_csv(path):
    if not path.exists():
        return None, f"missing source file: {rel_path(path)}"
    try:
        return pd.read_csv(path), None
    except Exception as exc:
        return None, f"read error: {exc}"


def dataframe_records(df, source_file, record_type):
    records = []
    exported_at = utc_now()
    for row in df.to_dict(orient="records"):
        records.append(sanitize_document({
            "source_file": rel_path(source_file),
            "record_type": record_type,
            "exported_at": exported_at,
            **row,
        }))
    return records


def business_summary_records(df, source_file):
    records = []
    exported_at = utc_now()
    for row in df.to_dict(orient="records"):
        records.append(sanitize_document({
            "source_file": rel_path(source_file),
            "record_type": "business_value_summary",
            "exported_at": exported_at,
            "metric_name": row.get("item"),
            "metric_value": row.get("value"),
        }))
    return records


def summary_row(source_file, collection, attempted, inserted, status, notes):
    return {
        "source_file": rel_path(source_file),
        "collection": collection,
        "records_attempted": attempted,
        "records_inserted": inserted,
        "status": status,
        "notes": notes,
    }


def source_definitions(config):
    return [
        {
            "path": MLFLOW_SUMMARY_PATH,
            "collection": config["experiments_collection"],
            "record_type": "mlflow_tracking_summary",
            "builder": dataframe_records,
        },
        {
            "path": BUSINESS_VALUE_SUMMARY_PATH,
            "collection": config["business_value_collection"],
            "record_type": "business_value_summary",
            "builder": business_summary_records,
        },
        {
            "path": BUSINESS_VALUE_SCENARIOS_PATH,
            "collection": config["business_value_collection"],
            "record_type": "business_value_scenario",
            "builder": dataframe_records,
        },
    ]


def build_documents(source):
    df, error = read_csv(source["path"])
    if error:
        return [], error
    if df is None or df.empty:
        return [], "source file is empty"

    if source["builder"] is business_summary_records:
        return source["builder"](df, source["path"]), None
    return source["builder"](df, source["path"], source["record_type"]), None


def write_export_summary(rows):
    EXPORT_SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows, columns=[
        "source_file",
        "collection",
        "records_attempted",
        "records_inserted",
        "status",
        "notes",
    ]).to_csv(EXPORT_SUMMARY_PATH, index=False)


def skipped_without_uri(config):
    rows = []
    for source in source_definitions(config):
        documents, note = build_documents(source)
        note_parts = ["MONGODB_URI is not configured"]
        if note:
            note_parts.append(note)
        rows.append(summary_row(
            source["path"],
            source["collection"],
            len(documents),
            0,
            "skipped",
            " | ".join(note_parts),
        ))
    write_export_summary(rows)
    return rows


def export_to_mongodb(config):
    rows = []
    client = get_mongo_client(config["uri"])
    ping_mongodb(client)

    for source in source_definitions(config):
        documents, note = build_documents(source)
        if note:
            rows.append(summary_row(source["path"], source["collection"], 0, 0, "skipped", note))
            continue

        collection = get_collection(client, config["db_name"], source["collection"])
        try:
            inserted_ids = insert_many_documents(collection, documents)
            rows.append(summary_row(
                source["path"],
                source["collection"],
                len(documents),
                len(inserted_ids),
                "success",
                "inserted",
            ))
        except Exception as exc:
            rows.append(summary_row(
                source["path"],
                source["collection"],
                len(documents),
                0,
                "failed",
                str(exc),
            ))

    write_export_summary(rows)
    return rows


def main():
    config = load_mongodb_config_from_env()

    if not config["uri"]:
        rows = skipped_without_uri(config)
        print(f"MongoDB export skipped. Configure MONGODB_URI. Summary: {rel_path(EXPORT_SUMMARY_PATH)}")
        return rows

    try:
        rows = export_to_mongodb(config)
    except Exception as exc:
        rows = []
        for source in source_definitions(config):
            documents, note = build_documents(source)
            notes = str(exc)
            if note:
                notes = f"{notes} | {note}"
            rows.append(summary_row(
                source["path"],
                source["collection"],
                len(documents),
                0,
                "failed",
                notes,
            ))
        write_export_summary(rows)

    print(f"MongoDB export summary written to {rel_path(EXPORT_SUMMARY_PATH)}")
    return rows


if __name__ == "__main__":
    main()
