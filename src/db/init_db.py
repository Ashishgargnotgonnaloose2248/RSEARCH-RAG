from src.db.postgres import get_connection

def init_db():
    conn = get_connection()
    cur = conn.cursor()

    with open("src/db/schema.sql", "r") as f:
        cur.execute(f.read())

    conn.commit()
    cur.close()
    conn.close()

    print("Tables created successfully ✅")

if __name__ == "__main__":
    init_db()
