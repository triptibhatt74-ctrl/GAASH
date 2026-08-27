import os
import psycopg
from dotenv import load_dotenv

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")

if not DATABASE_URL:
    raise RuntimeError("DATABASE_URL not found.")

statements = [
    """
    ALTER TABLE conversations
    ADD CONSTRAINT fk_conversations_user
    FOREIGN KEY (user_id)
    REFERENCES users(id)
    ON DELETE CASCADE
    """,

    """
    ALTER TABLE conversation_messages
    ADD CONSTRAINT fk_messages_user
    FOREIGN KEY (user_id)
    REFERENCES users(id)
    ON DELETE CASCADE
    """,

    """
    ALTER TABLE conversation_messages
    ADD CONSTRAINT fk_messages_conversation
    FOREIGN KEY (conversation_id)
    REFERENCES conversations(conversation_id)
    ON DELETE CASCADE
    """,

    """
    ALTER TABLE screening_sessions
    ADD CONSTRAINT fk_screening_sessions_user
    FOREIGN KEY (user_id)
    REFERENCES users(id)
    ON DELETE CASCADE
    """,

    """
    ALTER TABLE screening_session_items
    ADD CONSTRAINT fk_screening_items_session
    FOREIGN KEY (session_id)
    REFERENCES screening_sessions(session_id)
    ON DELETE CASCADE
    """,

    """
    ALTER TABLE screening_measurements
    ADD CONSTRAINT fk_measurements_session
    FOREIGN KEY (session_id)
    REFERENCES screening_sessions(session_id)
    ON DELETE CASCADE
    """,

    """
    ALTER TABLE screening_measurements
    ADD CONSTRAINT fk_measurements_user
    FOREIGN KEY (user_id)
    REFERENCES users(id)
    ON DELETE CASCADE
    """,
]

with psycopg.connect(DATABASE_URL) as conn:
    with conn.cursor() as cur:
        for statement in statements:
            cur.execute(statement)

    conn.commit()

print("Foreign keys added successfully.")