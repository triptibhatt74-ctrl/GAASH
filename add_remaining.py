import os
import psycopg
from dotenv import load_dotenv

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")

if not DATABASE_URL:
    raise RuntimeError("DATABASE_URL not found.")

tables = [
    "emotion_records",
    "assessment_records",
    "check_ins",
    "user_profiles",
    "wellbeing_reports",
    "resource_favorites",
    "suggested_states",
    "functional_impairments",
    "sleep_reports",
    "weekly_summaries",
    "questionnaire_state",
    "risk_assessments",
    "recommendation_records",
    "follow_ups",
    "escalation_records",
]

with psycopg.connect(DATABASE_URL) as conn:
    with conn.cursor() as cur:

        # Safety check first
        for table in tables:
            cur.execute(
                f"""
                SELECT COUNT(*)
                FROM {table} t
                LEFT JOIN users u
                    ON u.id = t.user_id
                WHERE u.id IS NULL
                """
            )

            count = cur.fetchone()[0]

            if count:
                raise RuntimeError(
                    f"{table} contains {count} orphan user rows."
                )

        # Add missing user ownership FKs
        for table in tables:

            constraint = f"fk_{table}_user"

            cur.execute(
                """
                SELECT 1
                FROM pg_constraint
                WHERE conname = %s
                """,
                (constraint,),
            )

            if cur.fetchone():
                print(f"SKIP {constraint} - already exists")
                continue

            cur.execute(
                f"""
                ALTER TABLE {table}
                ADD CONSTRAINT {constraint}
                FOREIGN KEY (user_id)
                REFERENCES users(id)
                ON DELETE CASCADE
                """
            )

            print(f"ADDED {constraint}")

    conn.commit()

print("Remaining user foreign keys complete.")