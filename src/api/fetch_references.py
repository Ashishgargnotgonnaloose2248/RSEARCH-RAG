import requests
import time
import psycopg2
from dotenv import load_dotenv
import os

# Load API key
load_dotenv("src/config/.env")

API_KEY = os.getenv("S2_API_KEY")
if not API_KEY:
    raise ValueError("API key not found.")

BASE_URL = "https://api.semanticscholar.org/graph/v1/paper/"
headers = {"x-api-key": API_KEY}

# Connect to PostgreSQL
conn = psycopg2.connect(
    dbname="RagDb",
    user="postgres",
    password="Ashishgarg22#",
    host="localhost",
    port="5432"
)

cur = conn.cursor()


def fetch_references(max_refs_per_paper=30):

    # Fetch all seed papers
    cur.execute("SELECT paper_id FROM papers;")
    paper_ids = cur.fetchall()

    print(f"Total seed papers: {len(paper_ids)}")

    processed = 0

    for (paper_id,) in paper_ids:

        url = f"{BASE_URL}{paper_id}"
        params = {
            "fields": "references.paperId,references.title,references.abstract,references.year"
        }

        try:
            response = requests.get(
                url,
                headers=headers,
                params=params,
                timeout=15
            )
        except requests.RequestException as e:
            print(f"Request error for {paper_id}: {e}")
            time.sleep(5)
            continue

        # Handle rate limiting
        if response.status_code == 429:
            print("Rate limited. Sleeping 10 seconds...")
            time.sleep(10)
            continue

        if response.status_code >= 500:
            print(f"Server error {response.status_code}. Retrying...")
            time.sleep(5)
            continue

        if response.status_code != 200:
            print(f"Error {response.status_code} for {paper_id}")
            continue

        data = response.json()
        references = data.get("references") or []

        # 🔥 LIMIT references per paper
        references = references[:max_refs_per_paper]

        for ref in references:
            target_id = ref.get("paperId")
            title = ref.get("title")
            abstract = ref.get("abstract")
            year = ref.get("year")

            # Only insert if abstract exists (important for GNN stage)
            if target_id and abstract:

                # Insert referenced paper
                cur.execute("""
                    INSERT INTO papers (paper_id, title, abstract, year)
                    VALUES (%s, %s, %s, %s)
                    ON CONFLICT (paper_id) DO NOTHING;
                """, (target_id, title, abstract, year))

                # Insert citation edge
                cur.execute("""
                    INSERT INTO citations (citing_paper_id, cited_paper_id)
                    VALUES (%s, %s)
                    ON CONFLICT DO NOTHING;
                """, (paper_id, target_id))

        processed += 1

        # Commit every 20 papers (safer for large runs)
        if processed % 20 == 0:
            conn.commit()
            print(f"Committed at {processed} papers")

        print(f"Processed references for {paper_id}")
        time.sleep(1)  # Prevent API abuse

    conn.commit()
    print("All references processed and committed.")


if __name__ == "__main__":
    fetch_references(max_refs_per_paper=30)
    cur.close()
    conn.close()
    print("Reference expansion completed successfully.")