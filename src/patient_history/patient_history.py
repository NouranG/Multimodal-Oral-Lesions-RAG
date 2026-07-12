"""
patient_history.py
SQLite-based patient history manager.

Stores one record per consultation:
    - patient_id
    - timestamp
    - user_description
    - lesion_category + clinical_signals (from classify node)
    - generated_output
    - confidence_score
    - clarifying_question (if loop triggered)
    - had_image (bool — whether an image was uploaded)

Usage:
    from src.patient_history import PatientHistory
    db = PatientHistory()
    db.save_consultation(patient_id="P001", result=graph_result, had_image=True)
    records = db.get_patient_history("P001")
    all_ids = db.get_all_patient_ids()
    db.delete_patient("P001")
"""

import os
import json
import sqlite3
from datetime import datetime
from typing import Optional

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH  = os.path.join(BASE_DIR, "data", "patient_history.db")


class PatientHistory:

    def __init__(self, db_path: str = DB_PATH):
        self.db_path = db_path
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self._init_db()

    # ── Setup ─────────────────────────────────────────────────────────────────

    def _init_db(self):
        """Create tables if they don't exist yet."""
        with self._connect() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS consultations (
                    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
                    patient_id          TEXT    NOT NULL,
                    timestamp           TEXT    NOT NULL,
                    user_description    TEXT,
                    had_image           INTEGER DEFAULT 0,
                    lesion_category     TEXT,
                    clinical_signals    TEXT,   -- JSON string
                    refined_query       TEXT,
                    route_decision      TEXT,
                    generated_output    TEXT,
                    confidence_score    REAL,
                    clarifying_question TEXT,
                    retry_count         INTEGER DEFAULT 0
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_patient_id
                ON consultations (patient_id)
            """)

    def _connect(self) -> sqlite3.Connection:
        """Return a context-managed SQLite connection."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row   # rows behave like dicts
        return conn

    # ── Write ─────────────────────────────────────────────────────────────────

    def save_consultation(
        self,
        patient_id: str,
        result: dict,
        had_image: bool = False
    ) -> int:
        """
        Save one consultation to the database.
        result is the dict returned by graph.invoke().
        Returns the new row id.
        """
        signals = result.get("clinical_signals", {})
        signals_json = json.dumps(signals) if signals else "{}"

        row = {
            "patient_id":          patient_id.strip().upper(),
            "timestamp":           datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "user_description":    result.get("user_description") or "",
            "had_image":           1 if had_image else 0,
            "lesion_category":     result.get("lesion_category", ""),
            "clinical_signals":    signals_json,
            "refined_query":       result.get("refined_query", ""),
            "route_decision":      result.get("route_decision", ""),
            "generated_output":    result.get("generated_output", ""),
            "confidence_score":    result.get("confidence_score"),
            "clarifying_question": result.get("clarifying_question", ""),
            "retry_count":         result.get("retry_count", 0),
        }

        with self._connect() as conn:
            cursor = conn.execute("""
                INSERT INTO consultations (
                    patient_id, timestamp, user_description, had_image,
                    lesion_category, clinical_signals, refined_query,
                    route_decision, generated_output, confidence_score,
                    clarifying_question, retry_count
                ) VALUES (
                    :patient_id, :timestamp, :user_description, :had_image,
                    :lesion_category, :clinical_signals, :refined_query,
                    :route_decision, :generated_output, :confidence_score,
                    :clarifying_question, :retry_count
                )
            """, row)
            return cursor.lastrowid

    # ── Read ──────────────────────────────────────────────────────────────────

    def get_patient_history(self, patient_id: str) -> list[dict]:
        """
        Return all consultations for a patient, newest first.
        clinical_signals is deserialized from JSON back to a dict.
        """
        pid = patient_id.strip().upper()
        with self._connect() as conn:
            rows = conn.execute("""
                SELECT * FROM consultations
                WHERE patient_id = ?
                ORDER BY timestamp DESC
            """, (pid,)).fetchall()

        result = []
        for row in rows:
            record = dict(row)
            # Deserialize clinical_signals JSON string back to dict
            try:
                record["clinical_signals"] = json.loads(record["clinical_signals"] or "{}")
            except json.JSONDecodeError:
                record["clinical_signals"] = {}
            result.append(record)
        return result

    def get_consultation(self, consultation_id: int) -> Optional[dict]:
        """Return a single consultation by its row id."""
        with self._connect() as conn:
            row = conn.execute("""
                SELECT * FROM consultations WHERE id = ?
            """, (consultation_id,)).fetchone()
        if not row:
            return None
        record = dict(row)
        try:
            record["clinical_signals"] = json.loads(record["clinical_signals"] or "{}")
        except json.JSONDecodeError:
            record["clinical_signals"] = {}
        return record

    def get_all_patient_ids(self) -> list[str]:
        """Return sorted list of all unique patient IDs in the database."""
        with self._connect() as conn:
            rows = conn.execute("""
                SELECT DISTINCT patient_id FROM consultations
                ORDER BY patient_id ASC
            """).fetchall()
        return [row["patient_id"] for row in rows]

    def get_summary_stats(self, patient_id: str) -> dict:
        """
        Return summary stats for a patient:
        total consultations, categories seen, avg confidence score.
        """
        pid = patient_id.strip().upper()
        with self._connect() as conn:
            row = conn.execute("""
                SELECT
                    COUNT(*)                    AS total,
                    AVG(confidence_score)       AS avg_confidence,
                    GROUP_CONCAT(DISTINCT lesion_category) AS categories
                FROM consultations
                WHERE patient_id = ?
            """, (pid,)).fetchone()
        return dict(row) if row else {}

    # ── Delete ────────────────────────────────────────────────────────────────

    def delete_consultation(self, consultation_id: int):
        """Delete a single consultation record."""
        with self._connect() as conn:
            conn.execute("DELETE FROM consultations WHERE id = ?", (consultation_id,))

    def delete_patient(self, patient_id: str):
        """Delete ALL records for a patient — irreversible."""
        pid = patient_id.strip().upper()
        with self._connect() as conn:
            conn.execute("DELETE FROM consultations WHERE patient_id = ?", (pid,))

    # ── Export ────────────────────────────────────────────────────────────────

    def export_patient_json(self, patient_id: str) -> str:
        """Export all consultations for a patient as a JSON string."""
        records = self.get_patient_history(patient_id)
        return json.dumps(records, indent=2, ensure_ascii=False)
