import os
import psycopg
from dotenv import load_dotenv

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")

if not DATABASE_URL:
    raise RuntimeError("DATABASE_URL not found in .env")

checks = {
    "orphan_conversations": """
        SELECT COUNT(*)
        FROM conversations c
        LEFT JOIN users u ON u.id = c.user_id
        WHERE u.id IS NULL
    """,

    "orphan_messages_users": """
        SELECT COUNT(*)
        FROM conversation_messages m
        LEFT JOIN users u ON u.id = m.user_id
        WHERE u.id IS NULL
    """,

    "orphan_messages_conversations": """
        SELECT COUNT(*)
        FROM conversation_messages m
        LEFT JOIN conversations c
          ON c.conversation_id = m.conversation_id
        WHERE m.conversation_id IS NOT NULL
          AND c.conversation_id IS NULL
    """,

    "orphan_screening_sessions": """
        SELECT COUNT(*)
        FROM screening_sessions s
        LEFT JOIN users u ON u.id = s.user_id
        WHERE u.id IS NULL
    """,

    "orphan_screening_items": """
        SELECT COUNT(*)
        FROM screening_session_items i
        LEFT JOIN screening_sessions s
          ON s.session_id = i.session_id
        WHERE s.session_id IS NULL
    """,

    "orphan_measurements": """
        SELECT COUNT(*)
        FROM screening_measurements m
        LEFT JOIN screening_sessions s
          ON s.session_id = m.session_id
        WHERE s.session_id IS NULL
    """,

    "orphan_measurement_users": """
        SELECT COUNT(*)
        FROM screening_measurements m
        LEFT JOIN users u ON u.id = m.user_id
        WHERE u.id IS NULL
    """,
}

with psycopg.connect(DATABASE_URL) as conn:
    with conn.cursor() as cur:

        for name, query in checks.items():
            cur.execute(query)
            count = cur.fetchone()[0]

            print(f"{name}: {count}")