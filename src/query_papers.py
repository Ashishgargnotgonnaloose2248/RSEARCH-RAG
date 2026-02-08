from src.db.postgres import get_connection

conn = get_connection()
cur = conn.cursor()

cur.execute("SELECT * FROM papers;")
papers = cur.fetchall()

print("Papers in database:")
print("-" * 80)
for paper in papers:
    print(f"ID: {paper[0]}")
    print(f"Title: {paper[1]}")
    print(f"Abstract: {paper[2]}")
    print(f"Year: {paper[3]}")
    print("-" * 80)

print(f"\nTotal papers: {len(papers)}")

cur.close()
conn.close()
