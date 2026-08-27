import os
import psycopg
from dotenv import load_dotenv

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")

if not DATABASE_URL:
    raise RuntimeError("DATABASE_URL not found.")


relations = [
    # table, column, parent_table, parent_column, constraint, on_delete

    (
        "resource_favorites",
        "resource_id",
        "resources",
        "resource_id",
        "fk_resource_favorites_resource",
        "CASCADE",
    ),

    (
        "emotion_records",
        "conversation_id",
        "conversations",
        "conversation_id",
        "fk_emotion_records_conversation",
        "SET NULL",
    ),

    (
        "screening_sessions",
        "conversation_id",
        "conversations",
        "conversation_id",
        "fk_screening_sessions_conversation",
        "SET NULL",
    ),

    (
        "suggested_states",
        "conversation_id",
        "conversations",
        "conversation_id",
        "fk_suggested_states_conversation",
        "SET NULL",
    ),

    (
        "questionnaire_state",
        "conversation_id",
        "conversations",
        "conversation_id",
        "fk_questionnaire_state_conversation",
        "SET NULL",
    ),

    (
        "questionnaire_state",
        "session_id",
        "screening_sessions",
        "session_id",
        "fk_questionnaire_state_session",
        "SET NULL",
    ),

    (
        "escalation_records",
        "trigger_message_id",
        "conversation_messages",
        "id",
        "fk_escalations_trigger_message",
        "SET NULL",
    ),
]


with psycopg.connect(DATABASE_URL) as conn:
    with conn.cursor() as cur:

        # -----------------------------------
        # 1. Check for orphan references
        # -----------------------------------

        for (
            table,
            column,
            parent_table,
            parent_column,
            constraint,
            on_delete,
        ) in relations:

            cur.execute(
                f"""
                SELECT COUNT(*)
                FROM {table} child
                LEFT JOIN {parent_table} parent
                    ON parent.{parent_column}
                     = child.{column}
                WHERE child.{column} IS NOT NULL
                  AND parent.{parent_column} IS NULL
                """
            )

            count = cur.fetchone()[0]

            if count:
                raise RuntimeError(
                    f"{table}.{column} has "
                    f"{count} orphan references."
                )

        # -----------------------------------
        # 2. Add missing constraints
        # -----------------------------------

        for (
            table,
            column,
            parent_table,
            parent_column,
            constraint,
            on_delete,
        ) in relations:

            cur.execute(
                """
                SELECT 1
                FROM pg_constraint
                WHERE conname = %s
                """,
                (constraint,),
            )

            if cur.fetchone():
                print(
                    f"SKIP {constraint} - already exists"
                )
                continue

            cur.execute(
                f"""
                ALTER TABLE {table}
                ADD CONSTRAINT {constraint}
                FOREIGN KEY ({column})
                REFERENCES {parent_table}({parent_column})
                ON DELETE {on_delete}
                """
            )

            print(f"ADDED {constraint}")

    conn.commit()


print("Relationship foreign keys complete.")