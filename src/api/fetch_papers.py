import requests
import time
from dotenv import load_dotenv
import os

load_dotenv("src/config/.env")

API_KEY = os.getenv("S2_API_KEY")
if not API_KEY:
    raise ValueError("API key not found. Check your .env file.")

BASE_URL = "https://api.semanticscholar.org/graph/v1/paper/search"

headers = {
    "x-api-key": API_KEY
}


def fetch_cs_papers(limit=100):
    papers = []
    batch_size = 50  # Maximum safe batch size

    params = {
        "query": "machine learning OR computer systems OR databases",
        "fields": "paperId,title,abstract,year,citationCount,referenceCount,fieldsOfStudy"
    }

    for offset in range(0, limit, batch_size):

        params["limit"] = batch_size
        params["offset"] = offset

        print(f"\nFetching batch starting at offset {offset}...")

        try:
            response = requests.get(
                BASE_URL,
                headers=headers,
                params=params,
                timeout=30  # prevents hanging
            )
        except requests.exceptions.RequestException as e:
            print("Connection error:", e)
            print("Sleeping 15 seconds before retry...")
            time.sleep(15)
            continue

        # Handle rate limiting
        if response.status_code == 429:
            print("Rate limited (429). Sleeping 15 seconds...")
            time.sleep(15)
            continue

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

        print(f"Batch fetched successfully. Current total: {len(papers)}")

        time.sleep(3)  # important when scaling

    return papers


if __name__ == "__main__":
    results = fetch_cs_papers(limit=100)
    print(f"\nTotal papers fetched: {len(results)}")