import requests
import time
import psycopg2
from dotenv import load_dotenv
import os

load_dotenv("src/config/.env")

API_KEY = os.getenv("S2_API_KEY")
if not API_KEY:
    raise ValueError("API key not found.")

BASE_URL = "https://api.semanticscholar.org/graph/v1/paper/"
headers = {"x-api-key": API_KEY}

conn = psycopg2.connect(
    dbname="RagDb",
    user="postgres",
    password="Ashishgarg22#",
    host="localhost",
    port="5432"
)

cur = conn.cursor()


def fetch_citations():
    cur.execute("""
    SELECT paper_id 
    FROM papers
    ORDER BY citation_count DESC NULLS LAST
    LIMIT 300;
""")
    paper_ids = cur.fetchall()

    print(f"Processing {len(paper_ids)} papers for incoming citations")

    for (paper_id,) in paper_ids:

        url = f"{BASE_URL}{paper_id}"
        params = {"fields": "citations.paperId,citations.title,citations.year"}

        try:
            response = requests.get(url, headers=headers, params=params, timeout=15)
        except requests.RequestException:
            time.sleep(5)
            continue

        if response.status_code == 429:
            print("Rate limited. Sleeping 10 seconds...")
            time.sleep(10)
            continue

        if response.status_code != 200:
            continue

        data = response.json()
        citations = data.get("citations") or []

        for citing in citations:
            source_id = citing.get("paperId")
            title = citing.get("title")
            year = citing.get("year")

            if source_id:

                # Insert citing paper if not exists
                cur.execute("""
                    INSERT INTO papers (paper_id, title, year)
                    VALUES (%s, %s, %s)
                    ON CONFLICT (paper_id) DO NOTHING;
                """, (source_id, title, year))

                # Insert edge (incoming citation)
                cur.execute("""
                    INSERT INTO citations (citing_paper_id, cited_paper_id)
                    VALUES (%s, %s)
                    ON CONFLICT DO NOTHING;
                """, (source_id, paper_id))

        print(f"Processed citations for {paper_id}")
        time.sleep(1)

    conn.commit()


if __name__ == "__main__":
    fetch_citations()
    cur.close()
    conn.close()
    print("Citation expansion completed.")