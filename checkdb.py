import bot

with bot.get_conn() as conn:
    tables = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table'"
    ).fetchall()

print("TABLES:")
for table in tables:
    print(table[0])