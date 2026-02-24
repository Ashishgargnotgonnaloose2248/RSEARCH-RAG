import psycopg2
from src.api.fetch_papers import fetch_cs_papers

conn = psycopg2.connect(
    dbname="RagDb",
    user="postgres",
    password="Ashishgarg22#",
    host="localhost",
    port="5432"
)

cur = conn.cursor()

papers = fetch_cs_papers(limit=100)

for p in papers:
    cur.execute("""
        INSERT INTO papers (paper_id, title, abstract, year, citation_count, reference_count)
        VALUES (%s, %s, %s, %s, %s, %s)
        ON CONFLICT (paper_id) DO NOTHING;
    """, (
        p.get("paperId"),
        p.get("title"),
        p.get("abstract"),
        p.get("year"),
        p.get("citationCount"),
        p.get("referenceCount")
    ))

conn.commit()
cur.close()
conn.close()

print("Papers inserted successfully.")