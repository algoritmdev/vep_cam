import sqlite3

DB_NAME = "admin_sequence.db"

def init_db():
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS sequence (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            person_name TEXT NOT NULL,
            order_num INTEGER NOT NULL
        )
    """)
    conn.commit()
    conn.close()

def save_sequence(names):
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute("DELETE FROM sequence")
    for idx, name in enumerate(names):
        c.execute("INSERT INTO sequence (person_name, order_num) VALUES (?, ?)", (name.strip(), idx))
    conn.commit()
    conn.close()

def load_sequence():
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute("SELECT person_name FROM sequence ORDER BY order_num ASC")
    rows = c.fetchall()
    conn.close()
    return [r[0] for r in rows]
