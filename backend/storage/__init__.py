"""
MEDORBY Backend — Storage Module
Contains hospital DB (SQLite) and federated learning aggregator.
"""
from .hospital_db import (
    init_db, store_consultation, store_report_record,
    log_federated_contribution, get_records, get_db_stats,
)

__all__ = [
    "init_db", "store_consultation", "store_report_record",
    "log_federated_contribution", "get_records", "get_db_stats",
]
