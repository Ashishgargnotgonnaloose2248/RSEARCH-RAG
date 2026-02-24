from src.db.postgres import get_connection

def init_db():
    conn = get_connection()
    cur = conn.cursor()

    with open("src/db/schema.sql", "r") as f:
        cur.execute(f.read())

    # Ensure new columns exist in case the table was created before schema update
    cur.execute(
        "ALTER TABLE papers ADD COLUMN IF NOT EXISTS citation_count INT DEFAULT 0;"
    )
    cur.execute(
        "ALTER TABLE papers ADD COLUMN IF NOT EXISTS reference_count INT DEFAULT 0;"
    )

    conn.commit()
    cur.close()
    conn.close()

    print("Tables created successfully ✅")

if __name__ == "__main__":
    init_db()
