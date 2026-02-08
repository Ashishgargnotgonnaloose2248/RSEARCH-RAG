from src.db.postgres import get_connection

conn = get_connection()
cur = conn.cursor()

cur.execute("""
INSERT INTO papers (paper_id, title, abstract, year)
VALUES ('P1', 'Test Paper', 'This is a test abstract', 2024)
ON CONFLICT DO NOTHING;
""")

conn.commit()
cur.close()
conn.close()

print("Dummy paper inserted ✅")
