# run_migrations.py

import os
from pathlib import Path

import psycopg
from dotenv import load_dotenv

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")

if not DATABASE_URL:
    raise RuntimeError("DATABASE_URL not found in .env")

FILES = [
    "0001_phase1_foundation_preflight.sql",
    "0001_phase1_foundation.sql",
    "0002_phase2_distress_intelligence_preflight.sql",
    "0002_phase2_distress_intelligence.sql",
    "0003_phase3_authorized_operations_preflight.sql",
    "0003_phase3_authorized_operations.sql",
    "0006_dssb_screening_constraints.sql",
    "0007_dssb_adult_eligibility.sql",
]

base = Path(__file__).resolve().parent

with psycopg.connect(DATABASE_URL) as conn:
    for filename in FILES:
        path = base / filename

        if not path.exists():
            raise FileNotFoundError(
                f"Migration file not found: {path}"
            )

        sql = path.read_text(encoding="utf-8")

        print(f"\nRUNNING: {filename}")

        try:
            with conn.cursor() as cur:
                cur.execute(sql)

            conn.commit()
            print(f"SUCCESS: {filename}")

        except Exception:
            conn.rollback()
            print(f"FAILED: {filename}")
            raise

print("\nALL MIGRATIONS COMPLETED.")
