import os
import psycopg
from dotenv import load_dotenv

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")

if not DATABASE_URL:
    raise RuntimeError("DATABASE_URL not found.")

with psycopg.connect(DATABASE_URL) as conn:
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT conname
            FROM pg_constraint
            WHERE conrelid = 'emotion_records'::regclass
              AND contype = 'c'
              AND pg_get_constraintdef(oid) ILIKE '%source%'
            """
        )

        row = cur.fetchone()

        if row:
            constraint_name = row[0]

            cur.execute(
                f"""
                ALTER TABLE emotion_records
                DROP CONSTRAINT {constraint_name}
                """
            )

            print(f"DROPPED {constraint_name}")

        cur.execute(
            """
            ALTER TABLE emotion_records
            ADD CONSTRAINT emotion_records_source_check
            CHECK (
                source IN (
                    'text',
                    'visual',
                    'voice'
                )
            )
            """
        )

        print("ADDED emotion_records_source_check")

    conn.commit()

print("Emotion source migration complete.")