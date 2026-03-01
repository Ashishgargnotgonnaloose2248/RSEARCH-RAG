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


def fetch_cs_papers(limit=800):
    papers = []
    batch_size = 50
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
            print("Retrying after 15 seconds...")
            time.sleep(15)
            continue

        if response.status_code == 429:
            print("Rate limited. Sleeping 15 seconds...")
            time.sleep(15)
            continue

        if response.status_code >= 500:
            print("Server error. Retrying...")
            time.sleep(10)
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
            if (
                paper.get("abstract") is not None
                and "Computer Science" in (paper.get("fieldsOfStudy") or [])
            ):
                papers.append(paper)

        print(f"Valid papers so far: {len(papers)}")

        offset += batch_size
        time.sleep(3)

    return papers


if __name__ == "__main__":
    results = fetch_cs_papers(limit=800)
    print(f"\nFinal total valid papers fetched: {len(results)}")