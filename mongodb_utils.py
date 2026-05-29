from __future__ import annotations

import math
import os
from datetime import date, datetime
from pathlib import Path
from typing import Any


DEFAULT_MONGODB_DB = "phishing_detection"
DEFAULT_EXPERIMENTS_COLLECTION = "experiment_results"
DEFAULT_BUSINESS_VALUE_COLLECTION = "business_value_runs"


def get_mongo_client(uri: str):
    """Create a MongoDB client from a connection URI."""
    from pymongo import MongoClient

    return MongoClient(uri, serverSelectionTimeoutMS=5000)


def get_collection(client, db_name: str, collection_name: str):
    """Return a MongoDB collection handle."""
    return client[db_name][collection_name]


def insert_one_document(collection, document):
    """Insert one sanitized document and return the inserted id."""
    result = collection.insert_one(sanitize_document(document))
    return result.inserted_id


def insert_many_documents(collection, documents):
    """Insert many sanitized documents and return inserted ids."""
    sanitized = [sanitize_document(document) for document in documents]
    if not sanitized:
        return []
    result = collection.insert_many(sanitized)
    return result.inserted_ids


def ping_mongodb(client):
    """Validate MongoDB connectivity."""
    return client.admin.command("ping")


def sanitize_document(document):
    """Convert common Python, pandas and numpy values to MongoDB-safe JSON data."""
    if isinstance(document, dict):
        return {str(key): sanitize_document(value) for key, value in document.items()}
    if isinstance(document, list):
        return [sanitize_document(value) for value in document]
    if isinstance(document, tuple):
        return [sanitize_document(value) for value in document]
    if isinstance(document, (datetime, date)):
        return document.isoformat()
    if isinstance(document, Path):
        return str(document)
    if document is None:
        return None

    try:
        import pandas as pd

        if pd.isna(document):
            return None
    except Exception:
        pass

    try:
        import numpy as np

        if isinstance(document, np.integer):
            return int(document)
        if isinstance(document, np.floating):
            value = float(document)
            return None if math.isnan(value) else value
        if isinstance(document, np.ndarray):
            return sanitize_document(document.tolist())
        if isinstance(document, np.bool_):
            return bool(document)
    except Exception:
        pass

    if isinstance(document, float) and math.isnan(document):
        return None
    return document


def load_mongodb_config_from_env():
    """Load MongoDB settings from environment variables and optional .env file."""
    try:
        from dotenv import load_dotenv

        load_dotenv()
    except Exception:
        pass

    return {
        "uri": os.getenv("MONGODB_URI"),
        "db_name": os.getenv("MONGODB_DB", DEFAULT_MONGODB_DB),
        "experiments_collection": os.getenv(
            "MONGODB_COLLECTION_EXPERIMENTS",
            DEFAULT_EXPERIMENTS_COLLECTION,
        ),
        "business_value_collection": os.getenv(
            "MONGODB_COLLECTION_BUSINESS_VALUE",
            DEFAULT_BUSINESS_VALUE_COLLECTION,
        ),
    }
