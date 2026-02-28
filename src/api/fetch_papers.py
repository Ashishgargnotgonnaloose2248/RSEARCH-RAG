import requests
import time
from dotenv import load_dotenv
import os

# Load API key
load_dotenv("src/config/.env")

API_KEY = os.getenv("S2_API_KEY")
if not API_KEY:
    raise ValueError("API key not found. Check your .env file.")

BASE_URL = "https://api.semanticscholar.org/graph/v1/paper/search"

headers = {
    "x-api-key": API_KEY
}


def fetch_cs_papers(limit=300):
    papers = []
    batch_size = 50  # Safe batch size for Semantic Scholar
    offset = 0

    params = {
        "query": "machine learning OR computer systems OR databases",
        "fields": "paperId,title,abstract,year,citationCount,referenceCount,fieldsOfStudy"
    }

    while offset < limit:

        params["limit"] = batch_size
        params["offset"] = offset

        print(f"\nFetching batch at offset {offset}...")

        try:
            response = requests.get(
                BASE_URL,
                headers=headers,
                params=params,
                timeout=30
            )
        except requests.exceptions.RequestException as e:
            print("Connection error:", e)
            print("Retrying same batch after 15 seconds...")
            time.sleep(15)
            continue  # retry same offset

        # Handle rate limiting
        if response.status_code == 429:
            print("Rate limited (429). Sleeping 15 seconds...")
            time.sleep(15)
            continue  # retry same offset

        # Handle server error
        if response.status_code >= 500:
            print("Server error:", response.status_code)
            print("Sleeping 10 seconds before retry...")
            time.sleep(10)
            continue  # retry same offset

        if response.status_code != 200:
            print("Error:", response.status_code)
            print(response.text)
            break

        data = response.json()
        batch = data.get("data", [])

        if not batch:
            print("No more papers returned.")
            break

        for paper in batch:
            if "Computer Science" in (paper.get("fieldsOfStudy") or []):
                papers.append(paper)

        print(f"Batch successful. Total papers so far: {len(papers)}")

        offset += batch_size
        time.sleep(3)  # Important delay for scaling

    return papers


if __name__ == "__main__":
    results = fetch_cs_papers(limit=300)
    print(f"\nFinal total papers fetched: {len(results)}")