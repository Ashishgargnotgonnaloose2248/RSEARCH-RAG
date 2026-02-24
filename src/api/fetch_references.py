import requests
import time
import psycopg2
from dotenv import load_dotenv
import os

load_dotenv("src/config/.env")

API_KEY = os.getenv("S2_API_KEY")

if not API_KEY:
    raise ValueError("API key not found. Check your .env file.")

BASE_URL = "https://api.semanticscholar.org/graph/v1/paper/"
headers = {"x-api-key": API_KEY}


# Connect to DB
conn = psycopg2.connect(
    dbname="RagDb",   # change if different
    user="postgres",         # change if different
    password="Ashishgarg22#",# change
    host="localhost",
    port="5432"
)

cur = conn.cursor()


def fetch_references(limit=5):
    # Get some paper_ids from DB
    cur.execute("SELECT paper_id FROM papers LIMIT %s;", (limit,))
    paper_ids = cur.fetchall()

    for (paper_id,) in paper_ids:
        url = f"{BASE_URL}{paper_id}"
        params = {"fields": "references.paperId"}

        # Retry on 429 (rate limit) with exponential backoff
        max_retries = 3
        backoff = 1
        response = None
        for attempt in range(max_retries):
            try:
                response = requests.get(url, headers=headers, params=params, timeout=10)
            except requests.RequestException as e:
                print(f"Request error for {paper_id}: {e}")
                time.sleep(backoff)
                backoff *= 2
                continue

            if response.status_code == 200:
                break
            if response.status_code == 429:
                # rate limited: wait and retry
                wait = backoff * (2 ** attempt)
                print(f"Rate limited (429). Sleeping {wait}s before retrying...")
                time.sleep(wait)
                continue
            if response.status_code == 404:
                print(f"Paper not found: {paper_id} (404)")
                break
            # other non-success codes: log and skip
            print("Error:", response.status_code)
            break

        if response is None or response.status_code != 200:
            continue

        data = response.json()
        references = data.get("references") or []

        for ref in references:
            target_id = ref.get("paperId")

            if target_id:
                cur.execute("""
                    INSERT INTO citations (citing_paper_id, cited_paper_id)
                    VALUES (%s, %s)
                    ON CONFLICT DO NOTHING;
                """, (paper_id, target_id))

        print(f"Processed references for {paper_id}")
        time.sleep(1)  # avoid rate limit

    conn.commit()


if __name__ == "__main__":
    fetch_references(limit=5)
    cur.close()
    conn.close()
    print("Reference extraction completed.")
